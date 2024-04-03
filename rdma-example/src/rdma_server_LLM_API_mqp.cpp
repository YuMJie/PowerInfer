#include "common.h"
#include "ggml.h"

#include <locale.h>
#include <assert.h>
#include <math.h>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <unordered_map>
#include <queue>
#include <string.h>
#include <cassert>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <vector>
#include "rdma_common.h"
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define NUM_QPS 5
//rdma
static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_server_id = NULL, *cm_client_id = NULL;
static struct rdma_cm_id *cm_client_ids[NUM_QPS];
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_comp_channel *io_completion_channels[NUM_QPS];

static struct ibv_cq *cq = NULL;
static struct ibv_cq *cqs[NUM_QPS];
static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp_init_attr qp_init_attrs[NUM_QPS];
static struct ibv_qp *client_qp = NULL;
static struct ibv_qp *client_qps[NUM_QPS];
/* RDMA memory resources */
static struct ibv_mr *client_metadata_mr = NULL, *server_buffer_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_recv_wr client_recv_wr, *bad_client_recv_wr = NULL;
static struct ibv_send_wr server_send_wr, *bad_server_send_wr = NULL;
static struct ibv_sge client_recv_sge, server_send_sge;

static struct rdma_buffer_attr_vec server_metadata_attrs;
static struct rdma_buffer_attr_vec client_metadata_attrs;
std::vector<struct ibv_mr *> server_buffer_mrs;


static int server_recv_buffer(ibv_mr *server_mr)
{
	int ret = -1;

	client_recv_sge.addr = (uint64_t) server_mr->addr;
	client_recv_sge.length = (uint32_t) server_mr->length;
	client_recv_sge.lkey = (uint32_t) server_mr->lkey;
	/* now we link it to the request */
	bzero(&client_recv_sge, sizeof(client_recv_sge)); //bzero函数将server_recv_wr结构体清零，并将server_recv_sge结构体的地址赋值给server_recv_wr的sg_list成员，将1赋值给server_recv_wr的num_sge成员。这些操作将接收缓冲区的属性与请求相关联。
	client_recv_wr.sg_list = &client_recv_sge;
	client_recv_wr.num_sge = 1;
	ret = ibv_post_recv(client_qps[2] /* which QP */, //代码调用ibv_post_recv函数来提交接收工作请求。该函数接受一些参数，包括一个指向客户端QP（Queue Pair）的指针、一个指向接收工作请求的指针以及一个指向错误工作请求的指针。如果提交成功，函数将返回0，否则返回一个非零值
		      &client_recv_wr /* receive work request*/,
		      &bad_client_recv_wr /* error WRs */);
	if (ret) {
		rdma_error("Failed to pre-post the receive buffer, errno: %d \n", ret);
		return ret;
	}
	debug("Receive buffer pre-posting is successful \n");
	return 0;
}


static int start_rdma_server_all(struct sockaddr_in *server_addr) 
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
	ret = rdma_listen(cm_server_id, 0); /* backlog = 8 clients, same as TCP, see man listen*/ //开始监听传入的 IP 和端口。并指定最大连接数（这里是 8）。
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
	for(int i = 0; i < NUM_QPS; i++) {
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
	cm_client_ids[i] = cm_event->id; //获取新连接的客户端标识符 
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
	debug("A new RDMA client connection id is stored at %p\n", cm_client_ids[i]);


	if(!cm_client_ids[i]){
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
	if(!i)	pd = ibv_alloc_pd(cm_client_ids[0]->verbs  //这行代码使用ibv_alloc_pd函数为客户端连接分配一个保护域（Protection Domain）
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
	io_completion_channels[i] = ibv_create_comp_channel(cm_client_ids[i]->verbs);//ibv_create_comp_channel函数创建一个完成通道（Completion Channel），用于接收I/O完成事件的通知。完成通道与RDMA设备相关联，因此我们使用cm_client_id->verbs来指定RDMA设备。
	if (!io_completion_channels[i]) {
		rdma_error("Failed to create an I/O completion event channel, %d\n",
				-errno);
		return -errno;
	}
	debug("An I/O completion event channel is created at %p \n", 
			io_completion_channels[i]);
	/* Now we create a completion queue (CQ) where actual I/O 
	 * completion metadata is placed. The metadata is packed into a structure 
	 * called struct ibv_wc (wc = work completion). ibv_wc has detailed 
	 * information about the work completion. An I/O request in RDMA world 
	 * is called "work" ;) 
	 */
	cqs[i]= ibv_create_cq(cm_client_ids[i]->verbs /* which device*/,  //ibv_create_cq函数创建一个完成队列（Completion Queue），用于存储RDMA操作完成的通知。完成队列中的元数据被打包到一个叫做struct ibv_wc的结构体中，它包含了有关工作完成的详细信息。在RDMA世界中，一个I/O请求被称为“工作”。
			CQ_CAPACITY /* maximum capacity*/, 
			NULL /* user context, not used here */,
			io_completion_channels[i] /* which IO completion channel */, 
			0 /* signaling vector, not used here*/);
	if (!cqs[i]) {
		rdma_error("Failed to create a completion queue (cq), errno: %d\n",
				-errno);
		return -errno;
	}
	debug("Completion queue (CQ) is created at %p with %d elements \n", 
			cqs[i], cqs[i]->cqe);
	/* Ask for the event for all activities in the completion queue*/
	ret = ibv_req_notify_cq(cqs[i] /* on which CQ */,  //这行代码使用ibv_req_notify_cq函数请求在完成队列上接收所有活动的通知。这里的cq是指定的完成队列，0表示接收所有类型的事件通知，没有过滤。
			0 /* 0 = all event type, no filter*/);
	if (ret) {
		rdma_error("Failed to request notifications on CQ errno: %d \n",
				-errno);
		return -errno;
	}
	/* Now the last step, set up the queue pair (send, recv) queues and their capacity.
	 * The capacity here is define statically but this can be probed from the 
	 * device. We just use a small number as defined in rdma_common.h */ //
       bzero(&qp_init_attrs[i], sizeof qp_init_attrs[i]); //这一系列代码用于初始化一个队列对（Queue Pair）的属性。队列对是RDMA通信的基本单元，它包含了发送和接收数据的队列。在这里，我们设置了队列对的最大接收和发送工作请求数量，以及队列对的类型（这里是可靠连接类型）和关联的完成队列。
       qp_init_attrs[i].cap.max_recv_sge = MAX_SGE; /* Maximum SGE per receive posting */
       qp_init_attrs[i].cap.max_recv_wr = MAX_WR; /* Maximum receive posting capacity */
       qp_init_attrs[i].cap.max_send_sge = MAX_SGE; /* Maximum SGE per send posting */
       qp_init_attrs[i].cap.max_send_wr = MAX_WR; /* Maximum send posting capacity */
       qp_init_attrs[i].qp_type = IBV_QPT_RC; /* QP type, RC = Reliable connection */
       /* We use same completion queue, but one can use different queues */
       qp_init_attrs[i].recv_cq = cqs[i]; /* Where should I notify for receive completion operations */
       qp_init_attrs[i].send_cq = cqs[i]; /* Where should I notify for send completion operations */
       /*Lets create a QP */
       ret = rdma_create_qp(cm_client_ids[i] /* which connection id */, //这行代码使用rdma_create_qp函数创建一个队列对（Queue Pair）。它需要指定连接ID、保护域和队列对的初始属性。函数执行成功后，队列对的引用将保存在client_qp变量中。
		       pd /* which protection domain*/,
		       &qp_init_attrs[i] /* Initial attributes */);
       if (ret) {
	       rdma_error("Failed to create QP due to errno: %d\n", -errno);
	       return -errno;
       }
       /* Save the reference for handy typing but is not required */
       client_qps[i] = cm_client_ids[i]->qp;
       debug("Client QP created at %p\n", client_qps[i]);


/* Starts an RDMA server by allocating basic connection resources */

/* Pre-posts a receive buffer and accepts an RDMA client connection */

	struct rdma_conn_param conn_param;
	// struct rdma_cm_event *cm_event = NULL;
	struct sockaddr_in remote_sockaddr; 
	if(!cm_client_ids[i] || !client_qps[i]) {
		rdma_error("Client resources are not properly setup\n");
		return -EINVAL;
	}
	
       memset(&conn_param, 0, sizeof(conn_param)); //我们准备一个连接参数结构体conn_param，并将其初始化为零。这个结构体用于指定连接的一些参数，比如我们可以设置期望的请求深度（initiator_depth）和响应方资源数（responder_resources）
       /* this tell how many outstanding requests can we handle */
       conn_param.initiator_depth = 15; /* For this exercise, we put a small number here */
       /* This tell how many outstanding requests we expect other side to handle */
       conn_param.responder_resources = 15; /* For this exercise, we put a small number */
	   //rdma_accept函数接受客户端的连接请求，并传入连接参数。
	printf("cm_client_ids[i]:%p\n",cm_client_ids[i]);
       ret = rdma_accept(cm_client_ids[i], &conn_param);
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
	// memcpy(&remote_sockaddr /* where to save */, 
	// 		rdma_get_peer_addr(cm_client_ids[i]) /* gives you remote sockaddr */, 
	// 		sizeof(struct sockaddr_in) /* max size */);
	printf("A new connection is accepted from %s \n", 
			inet_ntoa(remote_sockaddr.sin_addr));
	}
	return ret;
}





ibv_mr *  register_mrs_to_client(std::vector<ggml_tensor*> tensor_dsts )  //该函数用于向连接的客户端发送服务器端缓冲区的元数据。
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
			}
			server_metadata_attrs.address[i] = (uint64_t) server_buffer_mrs[i]->addr;
			server_metadata_attrs.length[i] = (uint32_t) server_buffer_mrs[i]->length;
			server_metadata_attrs.stags[i].local_stag = (uint32_t) server_buffer_mrs[i]->lkey;
			
		}
			server_metadata_attrs.size = size;
       /* This buffer is used to transmit information about the above 
	* buffer to the client. So this contains the metadata about the server 
	* buffer. Hence this is called metadata buffer. Since this is already 
	* on allocated, we just register it. 
        * We need to prepare a send I/O operation that will tell the 
	* client the address of the server buffer. 
	*/
	//代码准备一个发送操作，用于告知客户端服务器端缓冲区的地址。代码将服务器端缓冲区的地址、长度和本地标签信息填充到server_metadata_attr 结构体中
       ibv_mr * server_metadata_mr = rdma_buffer_register(pd /* which protection domain*/,  //调用 rdma_buffer_register() 函数将其注册到保护域中
		       &server_metadata_attrs /* which memory to register */, 
		       sizeof(server_metadata_attrs) /* what is the size of memory */,
		       IBV_ACCESS_LOCAL_WRITE /* what access permission */);
       if(!server_metadata_mr){
	       rdma_error("Server failed to create to hold server metadata \n");
	       /* we assume that this is due to out of memory error */
       }
       /* We need to transmit this buffer. So we create a send request. 
	* A send request consists of multiple SGE elements. In our case, we only
	* have one 
	*/
		//代码创建一个发送请求，并将 server_metadata_attr 结构体的信息填充到 server_send_sge 结构体中。接着，代码将 server_send_sge 结构体与发送请求关联，并设置发送请求的操作码为 IBV_WR_SEND，表示这是一个发送请求。代码还设置发送请求的标志为 IBV_SEND_SIGNALED，表示希望接收到发送完成的通知。
	   server_recv_buffer(server_metadata_mr);
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
		ret = ibv_post_send(client_qps[3] /* which QP */,   
		&server_send_wr /* Send request that we prepared before */, 
		&bad_server_send_wr /* In case of error, this will contain failed requests */);
		if (ret) {
			rdma_error("Posting of server metdata failed, errno: %d \n",
					-errno);
		}
	   
	   //代码调用 ibv_post_send() 函数将发送请求提交到客户端的队列对列（QP）中，并检查是否提交成功。

       /* We check for completion notification */
       ret = process_work_completion_events(io_completion_channels[3], &wc, 1);
		debug("Local buffer metadata has been sent to the client \n");
		
		printf("wait writer \n");
		ret = process_work_completion_events(io_completion_channels[2], &wc, 1);
		// sleep(5);
		debug("Local buffer metadata has been sent to the client \n");
		return  server_metadata_mr;

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
	// rdma_buffer_deregister(server_metadata_mr);	 //并调用 rdma_buffer_deregister 函数来注销客户端和服务器的元数据内存资源
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





static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static float tensor_sum_elements(const ggml_tensor * tensor) {
    double sum = 0;
    if (tensor->type == GGML_TYPE_F32) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[0]; k++) {
                sum += ((float *) tensor->data)[j*tensor->ne[0] + k];
            }
        }
    }
    return sum;
}

static void tensor_dump(const ggml_tensor * tensor, const char * name) {
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi) - ", name,
        tensor->type, ggml_type_name(tensor->type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = tensor_sum_elements(tensor);
    printf("Sum of tensor %s is %6.2f\n", name, sum);
}

#define TENSOR_DUMP(tensor) tensor_dump(tensor, #tensor)

struct benchmark_params_struct {
    int32_t n_threads     = 1;
    int32_t n_iterations  = 10;
};

void printf_value(ggml_tensor * tensor )
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
void printf_nb(ggml_tensor * tensor )
{
    int ne0=tensor->nb[0];
    int ne1=tensor->nb[1];
    int ne2=tensor->nb[2];
    int ne3=tensor->nb[3];
    printf("nb: %i %i %i %i\n",ne0,ne1,ne2,ne3);
}

// void gpu_to_host(ggml_tensor * tensor)
// {   int g_main_device =0 ;
//     size_t size = tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3]*ggml_type_sizef(tensor->type);
//     cudaMemcpyAsync(tensor->data, tensor->extra[0],size, cudaMemcpyDeviceToHost);

// }

void printf_set(ggml_tensor * tensor)
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
                 void * data   = (char *) tensor->data + l*tensor->nb[0] + k*tensor->nb[1] + j*tensor->nb[2] + i*tensor->nb[3];
                int idx =l*tensor->nb[0] + k*tensor->nb[1] + j*tensor->nb[2] + i*tensor->nb[3];
                idx /=4;
                switch (tensor->type) {
                    case GGML_TYPE_I8:
                         ((int8_t *) data)[0]=idx;
                    case GGML_TYPE_I16:
                         ((int16_t *) data)[0]=idx;
                    case GGML_TYPE_I32:
                         ((int32_t *) data)[0]=idx;
                    // case GGML_TYPE_F16:
                    //     return GGML_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
                    case GGML_TYPE_F32:
                         ((float *) data)[0]=idx;
                         break;
                    default:
                        GGML_ASSERT(false);
                }
                }
            }
        }
    }
}
static void print_usage(int /*argc*/, char ** argv, struct benchmark_params_struct params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -i N, --iter N     number of iterations to use during computation (default: %d)\n", params.n_iterations);
    fprintf(stderr, "\n");
}

// void decode(struct ggml_cgraph * gf, struct ggml_tensor * embeddings, struct ggml_tensor * res,  llama_context &  lctx)
// {
//     #ifdef GGML_USE_CUBLAS
//     for (int i = 0; i < gf->n_leafs; i++) {
//         ggml_tensor * node = gf->leafs[i];
//         if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
//             ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char *) lctx.buf_alloc.data);
//             ggml_cuda_copy_to_device(node);
//         }
//     }

//     for (int i = 0; i < gf->n_nodes; i++) {
//         ggml_tensor * node = gf->nodes[i];
//         if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
//             ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char *) lctx.buf_alloc.data);
//         }
//     }

//     // HACK: ggml-alloc may change the tensor backend when reusing a parent, so force output to be on the CPU here if needed
//     if (!lctx.embedding.empty()) {
//         embeddings->backend = GGML_BACKEND_CPU;
//     }
//     res->backend = GGML_BACKEND_CPU;
// #endif
// }
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

int main(int argc, char ** argv)  {


    //RDMA
    char *ip = NULL, *port = NULL;
    int ret = -1;
    std::vector<struct ggml_tensor *> tensor_dsts;

	struct sockaddr_in server_sockaddr=get_server_sockaddr(ip, port);
	// ret = start_rdma_server(&server_sockaddr);
	// ret = setup_client_resources();
	// ret = accept_client_connection();
	ret =start_rdma_server_all(&server_sockaddr);

    struct benchmark_params_struct benchmark_params;

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_iterations = std::stoi(argv[i]);
        }  else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, benchmark_params);
            exit(0);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, benchmark_params);
        exit(1);
    }

    print_build_info();
    printf("Starting Test\n");

    // create the ggml context
    struct ggml_context * ctx;
    //const int sizex = 4096;
    //const int sizey = 11008;

#undef VERBOSE_DEBUGGING
#ifndef VERBOSE_DEBUGGING
    const int sizey = 2;
    const int sizex = 3;
    const int sizez = 6;
#else
    /* Working - let's increase size */
    const int sizey = 1;
    const int sizex = (8*32);
    const int sizez = 1;

    /*const int sizey = 1;
    const int sizex = 3*(8*32);
    const int sizez = 1;*/
#endif

    //printf("Memsize required = %i\n", sizex*sizex);

    // TODO: perform the bench for all types or for a user specified type
    const ggml_type qtype = GGML_TYPE_Q4_1;

    size_t ctx_size = 0;
    ctx_size += sizex*sizey*ggml_type_sizef(GGML_TYPE_F32);
    ctx_size += sizex*sizey*ggml_type_sizef(GGML_TYPE_F32);
    ctx_size += sizex*sizez*ggml_type_sizef(GGML_TYPE_F32);
    ctx_size += sizex*sizey*ggml_type_sizef(qtype);
    ctx_size += sizex*sizey*ggml_type_sizef(qtype);
    ctx_size += sizex*sizey*ggml_type_sizef(GGML_TYPE_F32); // BLAS
    ctx_size += sizex*sizey*ggml_type_sizef(GGML_TYPE_F32); // BLAS
    ctx_size += 1024*1024*16;

    printf("Allocating Memory of size %zi bytes, %zi MB\n",ctx_size, (ctx_size/1024/1024));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ 0
    };

    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }
    

    printf("Creating new tensors\n");
    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    // ggml_set_backend(m11, GGML_BACKEND_GPU);
    // ggml_cuda_transform_tensor(m11->data, m11);

    ggml_set_f32(m11, 41.5f);
    printf_nb(m11);
    // printf_set(m11);
    printf_value(m11);
	tensor_dsts.push_back(m11);

    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizey, sizez);
    // ggml_set_backend(m12, GGML_BACKEND_GPU);
    // ggml_cuda_transform_tensor(m12->data, m12);
    ggml_set_f32(m12, 1.5f);
    printf_nb(m12);
    printf_set(m12);
    printf_value(m12);

    struct ggml_tensor * m13 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizez, sizez);
    // ggml_set_backend(m13, GGML_BACKEND_GPU);
    // ggml_cuda_transform_tensor(m13->data, m13);
    ggml_set_f32(m13, 1.5f);
    printf_nb(m13);
    printf_set(m13);
    printf_value(m13);


    // printf("Creating new tensor m2\n");
    struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);

    ggml_set_f32(m2, 2.0f);
    printf_nb(m2);
    printf_set(m2);
    printf_value(m2);

    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);
    printf("m11xm2_1\n");

    //将输出发送给server
    struct ggml_tensor * m11xm2_1 = ggml_mul_mat(ctx, m11xm2, m12);
    //得到server的输出后继续计算
    printf("m11xm2_2\n");

    // tensor_dsts.push_back(m11xm2_1);
    struct ggml_tensor * m11xm2_2 = ggml_mul_mat(ctx, m11xm2_1, m13);
    // tensor_dsts.push_back(m11xm2_2);

    // ggml_set_backend(m11xm2, GGML_BACKEND_GPU);

    // printf("Creating compute graph\n");
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m11xm2_2);

    printf("n_threads=%i\n", benchmark_params.n_threads);

    TENSOR_DUMP(m11);
    TENSOR_DUMP(m2);



    //RDMA


    std::vector<uint8_t> work_buffer;

    ggml_graph_compute_helper(work_buffer, gf, benchmark_params.n_threads);
	ibv_mr *  server_metadata_mr = register_mrs_to_client(tensor_dsts);
	
	// ret = disconnect_and_cleanup_LLM_vec();
    
    printf("n_nodes=%i\n",gf->n_nodes);
    printf("n_leafs=%i\n",gf->n_leafs);
    for(int i=0;i< gf->n_nodes;i++)
    {
        printf("node %i\n",i);
        TENSOR_DUMP(gf->nodes[i]);
    }


    return 0;
}
