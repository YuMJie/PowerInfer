/*
 * This is a RDMA server side code. 
 *
 * Author: Animesh Trivedi 
 *         atrivedi@apache.org 
 *
 * TODO: Cleanup previously allocated resources in case of an error condition
 */
#include<stdio.h>
#include "rdma_common.h"
#include "string.h"
#include "llama.h"
#include <vector>
/* These are the RDMA resources needed to setup an RDMA connection */
/* Event channel, where connection management (cm) related events are relayed */
static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_event_channel *cm_event_channel_1 = NULL;
static struct rdma_cm_id *cm_server_id = NULL, *cm_client_id = NULL;
static struct rdma_cm_id *cm_server_id_1 = NULL, *cm_client_id_1 = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_comp_channel *io_completion_channel_1 = NULL;
static struct ibv_cq *cq = NULL;
static struct ibv_cq *cq_1 = NULL;
static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp_init_attr qp_init_attr_1;
static struct ibv_qp *client_qp = NULL;
/* RDMA memory resources */
static struct ibv_mr *client_metadata_mr = NULL, *server_buffer_mr = NULL, *server_metadata_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_recv_wr client_recv_wr, *bad_client_recv_wr = NULL;
static struct ibv_send_wr server_send_wr, *bad_server_send_wr = NULL;
static struct ibv_sge client_recv_sge, server_send_sge;

static struct rdma_buffer_attr_vec server_metadata_attrs;
static struct rdma_buffer_attr_vec client_metadata_attrs;
std::vector<struct ibv_mr *> server_buffer_mrs;
int total=90;
struct ggml_context * ctx =NULL;

/* When we call this function cm_client_id must be set to a valid identifier.
 * This is where, we prepare client connection before we accept it. This 
 * mainly involve pre-posting a receive buffer to receive client side 
 * RDMA credentials
 */

void printf_value(struct ggml_tensor * tensor )
{
    int ne0=tensor->ne[0];
    int ne1=tensor->ne[1];
    int ne2=tensor->ne[2];
    int ne3=tensor->ne[3];
    for(int i=0;i<ne3;i++)
    {
        for(int j=0;j<ne2;j++)
        {
            for(int k=0;k<ne1;k++)
            {
                for(int l=0;l<ne0;l++)
                {
                    printf("%f ",ggml_get_f32_nd(tensor,l,k,j,i));
                }
            }

        }
    }
}

static int setup_client_resources() //该函数用于准备客户端连接之前的一些资源设置。
{
	int ret = -1;
	if(!cm_client_id){
		rdma_error("Client id is still NULL \n");
		return -EINVAL;
	}
	/* We have a valid connection identifier, lets start to allocate 
	 * resources. We need: 
	 * 1. Protection Domains (PD) //保护域（Protection Domains，PD）：保护域类似于操作系统中的“进程抽象”，它是一组资源的集合，所有的资源都与特定的保护域相关联。在这里，我们需要为客户端连接创建一个保护域。
	 * 2. Memory Buffers  //为了进行RDMA通信，我们需要为客户端连接分配内存缓冲区，用于存储发送和接收的数据
	 * 3. Completion Queues (CQ) //完成队列用于存储RDMA操作完成的通知。在这里，我们需要为客户端连接创建一个完成队列。
	 * 4. Queue Pair (QP) //队列对是RDMA通信的基本单元，它包含了发送和接收数据的队列。在这里，我们需要为客户端连接创建一个队列对。
	 * Protection Domain (PD) is similar to a "process abstraction" 
	 * in the operating system. All resources are tied to a particular PD. 
	 * And accessing recourses across PD will result in a protection fault.
	 */
	pd = ibv_alloc_pd(cm_client_id->verbs  //这行代码使用ibv_alloc_pd函数为客户端连接分配一个保护域（Protection Domain）
			/* verbs defines a verb's provider, 
			 * i.e an RDMA device where the incoming 
			 * client connection came */);
	if (!pd) {
		rdma_error("Failed to allocate a protection domain errno: %d\n",
				-errno);
		return -errno;
	}
	debug("A new protection domain is allocated at %p \n", pd);
	/* Now we need a completion channel, were the I/O completion 
	 * notifications are sent. Remember, this is different from connection 
	 * management (CM) event notifications. 
	 * A completion channel is also tied to an RDMA device, hence we will 
	 * use cm_client_id->verbs. 
	 */
	io_completion_channel = ibv_create_comp_channel(cm_client_id->verbs);//ibv_create_comp_channel函数创建一个完成通道（Completion Channel），用于接收I/O完成事件的通知。完成通道与RDMA设备相关联，因此我们使用cm_client_id->verbs来指定RDMA设备。
	if (!io_completion_channel) {
		rdma_error("Failed to create an I/O completion event channel, %d\n",
				-errno);
		return -errno;
	}
	debug("An I/O completion event channel is created at %p \n", 
			io_completion_channel);
	/* Now we create a completion queue (CQ) where actual I/O 
	 * completion metadata is placed. The metadata is packed into a structure 
	 * called struct ibv_wc (wc = work completion). ibv_wc has detailed 
	 * information about the work completion. An I/O request in RDMA world 
	 * is called "work" ;) 
	 */
	cq = ibv_create_cq(cm_client_id->verbs /* which device*/,  //ibv_create_cq函数创建一个完成队列（Completion Queue），用于存储RDMA操作完成的通知。完成队列中的元数据被打包到一个叫做struct ibv_wc的结构体中，它包含了有关工作完成的详细信息。在RDMA世界中，一个I/O请求被称为“工作”。
			CQ_CAPACITY /* maximum capacity*/, 
			NULL /* user context, not used here */,
			io_completion_channel /* which IO completion channel */, 
			0 /* signaling vector, not used here*/);
	if (!cq) {
		rdma_error("Failed to create a completion queue (cq), errno: %d\n",
				-errno);
		return -errno;
	}
	debug("Completion queue (CQ) is created at %p with %d elements \n", 
			cq, cq->cqe);
	/* Ask for the event for all activities in the completion queue*/
	ret = ibv_req_notify_cq(cq /* on which CQ */,  //这行代码使用ibv_req_notify_cq函数请求在完成队列上接收所有活动的通知。这里的cq是指定的完成队列，0表示接收所有类型的事件通知，没有过滤。
			0 /* 0 = all event type, no filter*/);
	if (ret) {
		rdma_error("Failed to request notifications on CQ errno: %d \n",
				-errno);
		return -errno;
	}
	/* Now the last step, set up the queue pair (send, recv) queues and their capacity.
	 * The capacity here is define statically but this can be probed from the 
	 * device. We just use a small number as defined in rdma_common.h */ //
       bzero(&qp_init_attr, sizeof qp_init_attr); //这一系列代码用于初始化一个队列对（Queue Pair）的属性。队列对是RDMA通信的基本单元，它包含了发送和接收数据的队列。在这里，我们设置了队列对的最大接收和发送工作请求数量，以及队列对的类型（这里是可靠连接类型）和关联的完成队列。
       qp_init_attr.cap.max_recv_sge = MAX_SGE; /* Maximum SGE per receive posting */
       qp_init_attr.cap.max_recv_wr = MAX_WR; /* Maximum receive posting capacity */
       qp_init_attr.cap.max_send_sge = MAX_SGE; /* Maximum SGE per send posting */
       qp_init_attr.cap.max_send_wr = MAX_WR; /* Maximum send posting capacity */
       qp_init_attr.qp_type = IBV_QPT_RC; /* QP type, RC = Reliable connection */
       /* We use same completion queue, but one can use different queues */
       qp_init_attr.recv_cq = cq; /* Where should I notify for receive completion operations */
       qp_init_attr.send_cq = cq; /* Where should I notify for send completion operations */
       /*Lets create a QP */
       ret = rdma_create_qp(cm_client_id /* which connection id */, //这行代码使用rdma_create_qp函数创建一个队列对（Queue Pair）。它需要指定连接ID、保护域和队列对的初始属性。函数执行成功后，队列对的引用将保存在client_qp变量中。
		       pd /* which protection domain*/,
		       &qp_init_attr /* Initial attributes */);
       if (ret) {
	       rdma_error("Failed to create QP due to errno: %d\n", -errno);
	       return -errno;
       }
       /* Save the reference for handy typing but is not required */
       client_qp = cm_client_id->qp;
       debug("Client QP created at %p\n", client_qp);
       return ret;
}

/* Starts an RDMA server by allocating basic connection resources */
static int start_rdma_server(struct sockaddr_in *server_addr) 
{
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
	/*  Open a channel used to report asynchronous communication event */
	cm_event_channel = rdma_create_event_channel(); // 创建一个用于报告异步通信事件的通道
	if (!cm_event_channel) {
		rdma_error("Creating cm event channel failed with errno : (%d)", -errno);
		return -errno;
	}
	debug("RDMA CM event channel is created successfully at %p \n", 
			cm_event_channel);
	/* rdma_cm_id is the connection identifier (like socket) which is used 
	 * to define an RDMA connection. 
	 */
	ret = rdma_create_id(cm_event_channel, &cm_server_id, NULL, RDMA_PS_TCP); //创建一个 RDMA 连接标识符 cm_server_id，用于定义一个 RDMA 连接。rdma_create_id() 函数使用指定的通道和传输协议（这里是 TCP）创建一个 RDMA 连接标识符。
	if (ret) {
		rdma_error("Creating server cm id failed with errno: %d ", -errno);
		return -errno;
	}
	debug("A RDMA connection id for the server is created \n");
	/* Explicit binding of rdma cm id to the socket credentials */
	ret = rdma_bind_addr(cm_server_id, (struct sockaddr*) server_addr); //rdma_bind_addr() 函数将 RDMA 连接标识符绑定到指定的地址。
	if (ret) {
		rdma_error("Failed to bind server address, errno: %d \n", -errno);
		return -errno;
	}
	debug("Server RDMA CM id is successfully binded \n");
	/* Now we start to listen on the passed IP and port. However unlike
	 * normal TCP listen, this is a non-blocking call. When a new client is 
	 * connected, a new connection management (CM) event is generated on the 
	 * RDMA CM event channel from where the listening id was created. Here we
	 * have only one channel, so it is easy. */
	ret = rdma_listen(cm_server_id, 8); /* backlog = 8 clients, same as TCP, see man listen*/ //开始监听传入的 IP 和端口。并指定最大连接数（这里是 8）。
	if (ret) {
		rdma_error("rdma_listen failed to listen on server address, errno: %d ",
				-errno);
		return -errno;
	}
	printf("Server is listening successfully at: %s , port: %d \n",
			inet_ntoa(server_addr->sin_addr),
			ntohs(server_addr->sin_port));
	/* now, we expect a client to connect and generate a RDMA_CM_EVNET_CONNECT_REQUEST 
	 * We wait (block) on the connection management event channel for 
	 * the connect event. 
	 */
	ret = process_rdma_cm_event(cm_event_channel, 
			RDMA_CM_EVENT_CONNECT_REQUEST,
			&cm_event);
	if (ret) {
		rdma_error("Failed to get cm event, ret = %d \n" , ret);
		return ret;
	}
	/* Much like TCP connection, listening returns a new connection identifier 
	 * for newly connected client. In the case of RDMA, this is stored in id 
	 * field. For more details: man rdma_get_cm_event 
	 */
	cm_client_id = cm_event->id; //获取新连接的客户端标识符 
	/* now we acknowledge the event. Acknowledging the event free the resources 
	 * associated with the event structure. Hence any reference to the event 
	 * must be made before acknowledgment. Like, we have already saved the 
	 * client id from "id" field before acknowledging the event. 
	 */
	ret = rdma_ack_cm_event(cm_event); //函数确认接收到的连接管理事件，并释放与事件相关的资源。
	if (ret) {
		rdma_error("Failed to acknowledge the cm event errno: %d \n", -errno);
		return -errno;
	}
	debug("A new RDMA client connection id is stored at %p\n", cm_client_id);
	return ret;
}

/* Pre-posts a receive buffer and accepts an RDMA client connection */
static int accept_client_connection()
{
	struct rdma_conn_param conn_param;
	struct rdma_cm_event *cm_event = NULL;
	struct sockaddr_in remote_sockaddr; 
	int ret = -1;
	if(!cm_client_id || !client_qp) {
		rdma_error("Client resources are not properly setup\n");
		return -EINVAL;
	}
	
       memset(&conn_param, 0, sizeof(conn_param)); //我们准备一个连接参数结构体conn_param，并将其初始化为零。这个结构体用于指定连接的一些参数，比如我们可以设置期望的请求深度（initiator_depth）和响应方资源数（responder_resources）
       /* this tell how many outstanding requests can we handle */
       conn_param.initiator_depth = 15; /* For this exercise, we put a small number here */
       /* This tell how many outstanding requests we expect other side to handle */
       conn_param.responder_resources = 15; /* For this exercise, we put a small number */
	   //rdma_accept函数接受客户端的连接请求，并传入连接参数。
	   	   printf("cm_client_id:%p\n",cm_client_id);
       ret = rdma_accept(cm_client_id, &conn_param);
       if (ret) {
	       rdma_error("Failed to accept the connection, errno: %d \n", -errno);
	       return -errno;
       }
       /* We expect an RDMA_CM_EVNET_ESTABLISHED to indicate that the RDMA  
	* connection has been established and everything is fine on both, server 
	* as well as the client sides.
	*/	
        debug("Going to wait for : RDMA_CM_EVENT_ESTABLISHED event \n");
		//使用process_rdma_cm_event函数等待指定类型的RDMA CM事件，并将事件保存在cm_event变量中
       ret = process_rdma_cm_event(cm_event_channel, 
		       RDMA_CM_EVENT_ESTABLISHED,
		       &cm_event);
        if (ret) {
		rdma_error("Failed to get the cm event, errnp: %d \n", -errno);
		return -errno;
	}
	/* We acknowledge the event */
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge the cm event %d\n", -errno);
		return -errno;
	}
	//我们使用rdma_get_peer_addr函数获取连接的对端地址，并将其保存在remote_sockaddr变量中。
	/* Just FYI: How to extract connection information */
	memcpy(&remote_sockaddr /* where to save */, 
			rdma_get_peer_addr(cm_client_id) /* gives you remote sockaddr */, 
			sizeof(struct sockaddr_in) /* max size */);
	printf("A new connection is accepted from %s \n", 
			inet_ntoa(remote_sockaddr.sin_addr));
	return ret;
}

/* This function sends server side buffer metadata to the connected client */

static int send_server_metadata_to_client_LLM_vec()  //该函数用于向连接的客户端发送服务器端缓冲区的元数据。
{
	struct ibv_wc wc; //工作完成（work completion）结构体
	int ret = -1;
	/* Now, we first wait for the client to start the communication by 
	 * sending the server its metadata info. The server does not use it 
	 * in our example. We will receive a work completion notification for 
	 * our pre-posted receive request.
	 */
	// debug("Waiting for client's buffer information... \n");
	// ret = process_work_completion_events(io_completion_channel, &wc, 1); //process_work_completion_events() 函数等待客户端启动通信，并接收客户端发送的元数据信息。
	// if (ret != 1) {
	// 	rdma_error("Failed to receive , ret = %d \n", ret);
	// 	return ret;
	// }
	/* if all good, then we should have client's buffer information, lets see */
	printf("Client side buffer information is received...\n");
	// show_rdma_buffer_attr(&client_metadata_attrs.length[0]);
	printf("The client has requested buffer length of : %d bytes \n", 
			client_metadata_attrs.length[0]);
	size_t size = client_metadata_attrs.size;
	/* We need to setup requested memory buffer. This is where the client will 
	* do RDMA READs and WRITEs. */
//dma_buffer_alloc() 函数为服务器端分配内存缓冲区，并设置访问权限。
		server_buffer_mrs.resize(size);
		for(int i=0;i<size;++i)
		{
			   server_buffer_mrs[i] = rdma_buffer_alloc(pd /* which protection domain */, 
		       client_metadata_attrs.length[i] /* what size to allocate */, 
		       (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|
		       IBV_ACCESS_REMOTE_READ|
		       IBV_ACCESS_REMOTE_WRITE) /* access permissions */);

			if(!server_buffer_mrs[i]){
				rdma_error("Server failed to create a buffer \n");
				/* we assume that it is due to out of memory error */
				return -ENOMEM;
			}
			server_metadata_attrs.address[i] = (uint64_t) server_buffer_mrs[i]->addr;
			server_metadata_attrs.length[i] = (uint32_t) server_buffer_mrs[i]->length;
			server_metadata_attrs.stags[i].local_stag = (uint32_t) server_buffer_mrs[i]->lkey;
		}

       /* This buffer is used to transmit information about the above 
	* buffer to the client. So this contains the metadata about the server 
	* buffer. Hence this is called metadata buffer. Since this is already 
	* on allocated, we just register it. 
        * We need to prepare a send I/O operation that will tell the 
	* client the address of the server buffer. 
	*/
	//代码准备一个发送操作，用于告知客户端服务器端缓冲区的地址。代码将服务器端缓冲区的地址、长度和本地标签信息填充到server_metadata_attr 结构体中

       server_metadata_mr = rdma_buffer_register(pd /* which protection domain*/,  //调用 rdma_buffer_register() 函数将其注册到保护域中
		       &server_metadata_attrs /* which memory to register */, 
		       sizeof(server_metadata_attrs) /* what is the size of memory */,
		       IBV_ACCESS_LOCAL_WRITE /* what access permission */);
       if(!server_metadata_mr){
	       rdma_error("Server failed to create to hold server metadata \n");
	       /* we assume that this is due to out of memory error */
	       return -ENOMEM;
       }
       /* We need to transmit this buffer. So we create a send request. 
	* A send request consists of multiple SGE elements. In our case, we only
	* have one 
	*/
		//代码创建一个发送请求，并将 server_metadata_attr 结构体的信息填充到 server_send_sge 结构体中。接着，代码将 server_send_sge 结构体与发送请求关联，并设置发送请求的操作码为 IBV_WR_SEND，表示这是一个发送请求。代码还设置发送请求的标志为 IBV_SEND_SIGNALED，表示希望接收到发送完成的通知。

	   server_send_sge.addr = (uint64_t) &server_metadata_attrs;
       server_send_sge.length = sizeof(server_metadata_attrs);
       server_send_sge.lkey = server_metadata_mr->lkey;
       /* now we link this sge to the send request */
       bzero(&server_send_wr, sizeof(server_send_wr));
       server_send_wr.sg_list = &server_send_sge;
       server_send_wr.num_sge = 1; // only 1 SGE element in the array 
       server_send_wr.opcode = IBV_WR_SEND; // This is a send request 
       server_send_wr.send_flags = IBV_SEND_SIGNALED; // We want to get notification 
       /* This is a fast data path operation. Posting an I/O request */
	   	// sleep(50);
		ret = ibv_post_send(client_qp /* which QP */,   
		&server_send_wr /* Send request that we prepared before */, 
		&bad_server_send_wr /* In case of error, this will contain failed requests */);
		if (ret) {
			rdma_error("Posting of server metdata failed, errno: %d \n",
					-errno);
			return -errno;
		}
	   
	   //代码调用 ibv_post_send() 函数将发送请求提交到客户端的队列对列（QP）中，并检查是否提交成功。

       /* We check for completion notification */
       ret = process_work_completion_events(io_completion_channel, &wc, 1);
       if (ret != 1) {
	       rdma_error("Failed to send server metadata, ret = %d \n", ret);
	       return ret;
       }
       debug("Local buffer metadata has been sent to the client \n");
       return 0;
}



static int register_mrs_to_client(std::vector<ggml_tensor*> tensor_dsts )  //该函数用于向连接的客户端发送服务器端缓冲区的元数据。
{
	struct ibv_wc wc; //工作完成（work completion）结构体
	int ret = -1;

		size_t size =  tensor_dsts.size();
		server_buffer_mrs.resize(size);
		for(int i=0;i<size;++i)
		{
			   server_buffer_mrs[i] = rdma_buffer_register(pd /* which protection domain */, 
		       tensor_dsts[i]->data,
			   ggml_nbytes(tensor_dsts[i]) /* what size to allocate */, 
		       (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|
		       IBV_ACCESS_REMOTE_READ|
		       IBV_ACCESS_REMOTE_WRITE) /* access permissions */);
			if(!server_buffer_mrs[i]){
				rdma_error("Server failed to create a buffer \n");
				/* we assume that it is due to out of memory error */
				return -ENOMEM;
			}
			server_metadata_attrs.address[i] = (uint64_t) server_buffer_mrs[i]->addr;
			server_metadata_attrs.length[i] = (uint32_t) server_buffer_mrs[i]->length;
			server_metadata_attrs.stags[i].local_stag = (uint32_t) server_buffer_mrs[i]->lkey;
		
		}
		
       /* This buffer is used to transmit information about the above 
	* buffer to the client. So this contains the metadata about the server 
	* buffer. Hence this is called metadata buffer. Since this is already 
	* on allocated, we just register it. 
        * We need to prepare a send I/O operation that will tell the 
	* client the address of the server buffer. 
	*/
	//代码准备一个发送操作，用于告知客户端服务器端缓冲区的地址。代码将服务器端缓冲区的地址、长度和本地标签信息填充到server_metadata_attr 结构体中

       server_metadata_mr = rdma_buffer_register(pd /* which protection domain*/,  //调用 rdma_buffer_register() 函数将其注册到保护域中
		       &server_metadata_attrs /* which memory to register */, 
		       sizeof(server_metadata_attrs) /* what is the size of memory */,
		       IBV_ACCESS_LOCAL_WRITE /* what access permission */);
       if(!server_metadata_mr){
	       rdma_error("Server failed to create to hold server metadata \n");
	       /* we assume that this is due to out of memory error */
	       return -ENOMEM;
       }
       /* We need to transmit this buffer. So we create a send request. 
	* A send request consists of multiple SGE elements. In our case, we only
	* have one 
	*/
		//代码创建一个发送请求，并将 server_metadata_attr 结构体的信息填充到 server_send_sge 结构体中。接着，代码将 server_send_sge 结构体与发送请求关联，并设置发送请求的操作码为 IBV_WR_SEND，表示这是一个发送请求。代码还设置发送请求的标志为 IBV_SEND_SIGNALED，表示希望接收到发送完成的通知。

	   server_send_sge.addr = (uint64_t) &server_metadata_attrs;
       server_send_sge.length = sizeof(server_metadata_attrs);
       server_send_sge.lkey = server_metadata_mr->lkey;
       /* now we link this sge to the send request */
       bzero(&server_send_wr, sizeof(server_send_wr));
       server_send_wr.sg_list = &server_send_sge;
       server_send_wr.num_sge = 1; // only 1 SGE element in the array 
       server_send_wr.opcode = IBV_WR_SEND; // This is a send request 
       server_send_wr.send_flags = IBV_SEND_SIGNALED; // We want to get notification 
       /* This is a fast data path operation. Posting an I/O request */
	   	// sleep(50);
		ret = ibv_post_send(client_qp /* which QP */,   
		&server_send_wr /* Send request that we prepared before */, 
		&bad_server_send_wr /* In case of error, this will contain failed requests */);
		if (ret) {
			rdma_error("Posting of server metdata failed, errno: %d \n",
					-errno);
			return -errno;
		}
	   
	   //代码调用 ibv_post_send() 函数将发送请求提交到客户端的队列对列（QP）中，并检查是否提交成功。

       /* We check for completion notification */
       ret = process_work_completion_events(io_completion_channel, &wc, 1);

	   if (ret != 1) {
	       rdma_error("Failed to send server metadata, ret = %d \n", ret);
	       return ret;
       }
       debug("Local buffer metadata has been sent to the client \n");
       return 0;
}


/* This is server side logic. Server passively waits for the client to call 
 * rdma_disconnect() and then it will clean up its resources */
static int disconnect_and_cleanup_LLM_vec()
{	
	// sleep(1000);
	

	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
       /* Now we wait for the client to send us disconnect event */
       debug("Waiting for cm event: RDMA_CM_EVENT_DISCONNECTED\n");
       ret = process_rdma_cm_event(cm_event_channel,  //函数等待客户端发送断开连接事件。它调用了 process_rdma_cm_event 函数来处理 RDMA_CM_EVENT_DISCONNECTED 事件，并将返回的事件存储在 cm_event 中。
		       RDMA_CM_EVENT_DISCONNECTED, 
		       &cm_event);
       if (ret) {
	       rdma_error("Failed to get disconnect event, ret = %d \n", ret);
	       return ret;
       }
	/* We acknowledge the event */
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge the cm event %d\n", -errno);
		return -errno;
	}
	printf("A disconnect event is received from the client...\n");
	/* We free all the resources */
	/* Destroy QP */
	rdma_destroy_qp(cm_client_id); //首先，它销毁 QP（Queue Pair）。
	/* Destroy client cm id */
	ret = rdma_destroy_id(cm_client_id);//然后，销毁客户端的 cm id（Connection Manager Identifier）
	if (ret) {
		rdma_error("Failed to destroy client id cleanly, %d \n", -errno);
		// we continue anyways;
	}
	/* Destroy CQ */
	ret = ibv_destroy_cq(cq); //函数销毁完成通道（Completion Channel）。
	if (ret) {
		rdma_error("Failed to destroy completion queue cleanly, %d \n", -errno);
		// we continue anyways;
	}
	/* Destroy completion channel */
	ret = ibv_destroy_comp_channel(io_completion_channel);
	if (ret) {
		rdma_error("Failed to destroy completion channel cleanly, %d \n", -errno);
		// we continue anyways;
	}
	/* Destroy memory buffers */
	size_t size = server_buffer_mrs.size();
	for(int i=0;i<size;++i)
	{	
		// tensor_dsts[i]->data = server_buffer_mrs[i]->addr;
		// printf_value(tensor_dsts[i]); //函数释放内存缓冲区。它调用 rdma_buffer_free 函数来释放服务器缓冲区的内存资源，
		printf("\n");
	}

	for(int i=0;i<size;++i)
	{
		rdma_buffer_free(server_buffer_mrs[i]); //函数释放内存缓冲区。它调用 rdma_buffer_free 函数来释放服务器缓冲区的内存资源，
		// free(tensor_dsts[i]->data);
	}
	rdma_buffer_deregister(server_metadata_mr);	 //并调用 rdma_buffer_deregister 函数来注销客户端和服务器的元数据内存资源
	rdma_buffer_deregister(client_metadata_mr);	
	/* Destroy protection domain */
	ret = ibv_dealloc_pd(pd); //最后，函数销毁 rdma 服务器 id（Identifier）。
	if (ret) {
		rdma_error("Failed to destroy client protection domain cleanly, %d \n", -errno);
		// we continue anyways;
	}
	/* Destroy rdma server id */
	ret = rdma_destroy_id(cm_server_id); //最后，函数销毁 rdma 服务器 id（Identifier）。
	if (ret) {
		rdma_error("Failed to destroy server id cleanly, %d \n", -errno);
		// we continue anyways;
	}
	rdma_destroy_event_channel(cm_event_channel);
	printf("Server shut-down is complete \n");
	return 0;
}


void usage() 
{
	printf("Usage:\n");
	printf("rdma_server: [-a <server_addr>] [-p <server_port>]\n");
	printf("(default port is %d)\n", DEFAULT_RDMA_PORT);
	exit(1);
}

void func_get_ctx(){

    size_t ctx_size = 0;
    // ctx_size += 1000000*1000*ggml_type_sizef(GGML_TYPE_F32);
    ctx_size += 100*1024*16;

    printf("Allocating Memory of size %zi bytes, %zi MB\n",ctx_size, (ctx_size/1024/1024));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ 0
    };
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return ;
    }
}
sockaddr_in  get_server_sockaddr(char *ip, char * port) {
	struct sockaddr_in server_sockaddr;
	int ret, option;
	bzero(&server_sockaddr, sizeof server_sockaddr);
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if(ip)
	{
		ret = get_addr(ip, (struct sockaddr*) &server_sockaddr);
		if(ret) {
			rdma_error("Invalid IP \n");
			exit(1);
		}
	}
	if (!port) {
	  /* no port provided, use the default port */
	  server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);
	}
	else
	{	printf("port is %s\n",port);
	  server_sockaddr.sin_port = htons(strtol(port, NULL, 0)); 
	}

	return server_sockaddr;
}

int main(int argc, char **argv) 
{	func_get_ctx();
	char *ip = NULL, *port = NULL;
	// server_buffer_mrs.resize(total);
	std::vector<struct ggml_tensor *> tensor_dsts;

	tensor_dsts.resize(total);
	for(int i=0;i<total;++i)
	{	
		tensor_dsts[i] = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4);
		ggml_set_f32(tensor_dsts[i],3.4f);

	}
	int ret, option;


	/* Parse Command Line Arguments, not the most reliable code */
	while ((option = getopt(argc, argv, "a:p:")) != -1) {
		switch (option) {
			case 'a':
				/* Remember, this will overwrite the port info */
				ip = (optarg);
				break;
			case 'p':
				/* passed port to listen on */
				port = (optarg); 
				break;
			default:
				usage();
				break;
		}
	}

	struct sockaddr_in server_sockaddr=get_server_sockaddr(ip, port);
	ret = start_rdma_server(&server_sockaddr);
	if (ret) {
		rdma_error("RDMA server failed to start cleanly, ret = %d \n", ret);
		return ret;
	}
	ret = setup_client_resources();
	if (ret) { 
		rdma_error("Failed to setup client resources, ret = %d \n", ret);
		return ret;
	}
	ret = accept_client_connection();
	if (ret) {
		rdma_error("Failed to handle client cleanly, ret = %d \n", ret);
		return ret;
	}
	ret = register_mrs_to_client(tensor_dsts);
	if (ret) {
		rdma_error("Failed to send server metadata to the client, ret = %d \n", ret);
		return ret;
	}
	ret = disconnect_and_cleanup_LLM_vec();
	if (ret) { 
		rdma_error("Failed to clean up resources properly, ret = %d \n", ret);
		return ret;
	}
	return 0;
}
