/*
 * An example RDMA client side code. 
 * Author: Animesh Trivedi 
 *         atrivedi@apache.org
 */

#include "rdma_common.h"
#include  "llama.h"
// #include  "ggml.c"
#include  "ggml.h"
#include<vector>
/* These are basic RDMA resources */
/* These are RDMA connection related resources */
static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_client_id = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_cq *client_cq = NULL;
static struct ibv_qp *client_qp;
/* These are memory buffers related resources */
static struct ibv_mr *client_metadata_mr = NULL, 
		     *client_src_mr = NULL, 
		     *client_dst_mr = NULL, 
		     *server_metadata_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_send_wr client_send_wr, *bad_client_send_wr = NULL;
static struct ibv_recv_wr server_recv_wr, *bad_server_recv_wr = NULL;
static struct ibv_sge client_send_sge, server_recv_sge;
/* Source and Destination buffers, where RDMA operations source and sink */
// static struct rdma_buffer_attr_vec server_metadata_attrs;
static struct rdma_buffer_attr_vec client_metadata_attrs;


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

/* This is our testing function */
static int check_src_dst_LLM_vec_api(std::vector<ggml_tensor *> tensor_srcs,std::vector<ggml_tensor *> tensor_dsts) 
{			bool flag = true;
	size_t size = tensor_srcs.size();
	printf("%s\n",__func__);		
	for(int i=0;i<size;++i)
	{	printf("i=%d\n",i);
		printf_value(tensor_srcs[i]);
		printf_value(tensor_dsts[i]);
		if(memcmp((void*) tensor_srcs[i]->data, (void*) tensor_dsts[i]->data, ggml_nbytes(tensor_dsts[i]))!=0)
		{
			flag = false;
		}
	}
	return flag;

}
/* This function prepares client side connection resources for an RDMA connection */

/* Pre-posts a receive buffer before calling rdma_connect () */

/* Connects to the RDMA server */

void usage() {
	printf("Usage:\n");
	printf("rdma_client: [-a <server_addr>] [-p <server_port>] -s string (required)\n");
	printf("(default IP is 127.0.0.1 and port is %d)\n", DEFAULT_RDMA_PORT);
	exit(1);
}
struct ggml_context * ctx =NULL;
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


sockaddr_in  get_server_sockaddr(char *ip, char * port) 
{
	struct sockaddr_in server_sockaddr;
	int ret, option;
	bzero(&server_sockaddr, sizeof server_sockaddr);
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	ret = get_addr(ip, (struct sockaddr*) &server_sockaddr);
	if (!port) {
	  /* no port provided, use the default port */
	  server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);
	}
	else
	{	printf("port is %s\n",port);
	  server_sockaddr.sin_port = htons(strtol(port, NULL, 0)); 
	}
	if(ret) {
		rdma_error("Invalid IP \n");
		exit(1);
	}
	return server_sockaddr;
}
static int client_prepare_connection_api(struct sockaddr_in *s_addr)
{	
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
	/*  Open a channel used to report asynchronous communication event */
	cm_event_channel = rdma_create_event_channel();  //代码创建了一个RDMA事件通道
	if (!cm_event_channel) {
		rdma_error("Creating cm event channel failed, errno: %d ,.%s \n", -errno, strerror(errno));
		exit(1);
	}
	debug("RDMA CM event channel is created at : %p \n", cm_event_channel);
	/* rdma_cm_id is the connection identifier (like socket) which is used 
	 * to define an RDMA connection. 
	 */
	ret = rdma_create_id(cm_event_channel, &cm_client_id,  //函数创建一个RDMA连接标识符
			NULL,
			RDMA_PS_TCP);
	if (ret) {
		rdma_error("Creating cm id failed with errno: %d \n", -errno); 
		exit(1);
	}
	/* Resolve destination and optional source addresses from IP addresses  to
	 * an RDMA address.  If successful, the specified rdma_cm_id will be bound
	 * to a local device. */
	ret = rdma_resolve_addr(cm_client_id, NULL, (struct sockaddr*) s_addr, 2000); //函数将目标地址解析为RDMA地址，并将cm_client_id与本地设备绑定。
	if (ret) {
		rdma_error("Failed to resolve address, errno: %d \n", -errno);
		exit(1);
	}
	debug("waiting for cm event: RDMA_CM_EVENT_ADDR_RESOLVED\n");//process_rdma_cm_event函数等待并处理RDMA_CM_EVENT_ADDR_RESOLVED事件，该事件表示RDMA地址已解析完成
	ret  = process_rdma_cm_event(cm_event_channel, 
			RDMA_CM_EVENT_ADDR_RESOLVED,
			&cm_event);
	if (ret) {
		rdma_error("Failed to receive a valid event, ret = %d \n", ret);
		exit(1);
	}
	/* we ack the event */
	ret = rdma_ack_cm_event(cm_event); //确认接收到的事件
	if (ret) {
		rdma_error("Failed to acknowledge the CM event, errno: %d\n", -errno);
		exit(1);
	}
	debug("RDMA address is resolved \n");

	 /* Resolves an RDMA route to the destination address in order to 
	  * establish a connection */
	ret = rdma_resolve_route(cm_client_id, 2000); //函数解析到目标地址的RDMA路由，以建立连接
	if (ret) {
		rdma_error("Failed to resolve route, erno: %d \n", -errno);
	      exit(1);
	}
	debug("waiting for cm event: RDMA_CM_EVENT_ROUTE_RESOLVED\n");
	ret = process_rdma_cm_event(cm_event_channel, 
			RDMA_CM_EVENT_ROUTE_RESOLVED,
			&cm_event);
	if (ret) {
		rdma_error("Failed to receive a valid event, ret = %d \n", ret);
		exit(1);
	}
	/* we ack the event */
	ret = rdma_ack_cm_event(cm_event);//函数确认接收到的事件
	if (ret) {
		rdma_error("Failed to acknowledge the CM event, errno: %d \n", -errno);
		exit(1);
	}
	printf("Trying to connect to server at : %s port: %d \n", 
			inet_ntoa(s_addr->sin_addr),
			ntohs(s_addr->sin_port));
	/* Protection Domain (PD) is similar to a "process abstraction" 
	 * in the operating system. All resources are tied to a particular PD. 
	 * And accessing recourses across PD will result in a protection fault.
	 */
	pd = ibv_alloc_pd(cm_client_id->verbs);//函数分配一个保护域（Protection Domain），类似于操作系统中的"进程抽象"，所有资源都与特定的保护域相关联
	if (!pd) {
		rdma_error("Failed to alloc pd, errno: %d \n", -errno);
		exit(1);
	}
	debug("pd allocated at %p \n", pd);
	/* Now we need a completion channel, were the I/O completion 
	 * notifications are sent. Remember, this is different from connection 
	 * management (CM) event notifications. 
	 * A completion channel is also tied to an RDMA device, hence we will 
	 * use cm_client_id->verbs. 
	 */
	io_completion_channel = ibv_create_comp_channel(cm_client_id->verbs); //函数创建一个完成通道（Completion Channel），用于发送I/O完成通知
	if (!io_completion_channel) {
		rdma_error("Failed to create IO completion event channel, errno: %d\n",
			       -errno);
	exit(1);
	}
	debug("completion event channel created at : %p \n", io_completion_channel);
	/* Now we create a completion queue (CQ) where actual I/O 
	 * completion metadata is placed. The metadata is packed into a structure 
	 * called struct ibv_wc (wc = work completion). ibv_wc has detailed 
	 * information about the work completion. An I/O request in RDMA world 
	 * is called "work" ;) 
	 */
	client_cq = ibv_create_cq(cm_client_id->verbs /* which device*/,  //ibv_create_cq函数创建一个完成队列（Completion Queue），用于存放实际的I/O完成元数据
			CQ_CAPACITY /* maximum capacity*/, 
			NULL /* user context, not used here */,
			io_completion_channel /* which IO completion channel */, 
			0 /* signaling vector, not used here*/);
	if (!client_cq) {
		rdma_error("Failed to create CQ, errno: %d \n", -errno);
		exit(1);
	}
	debug("CQ created at %p with %d elements \n", client_cq, client_cq->cqe);
	ret = ibv_req_notify_cq(client_cq, 0); //ibv_req_notify_cq函数请求通知，以便在完成队列中有新的完成操作时得到通知
	if (ret) {
		rdma_error("Failed to request notifications, errno: %d\n", -errno);
		exit(1);
	}
	static struct ibv_qp_init_attr qp_init_attr;

       /* Now the last step, set up the queue pair (send, recv) queues and their capacity.
         * The capacity here is define statically but this can be probed from the 
	 * device. We just use a small number as defined in rdma_common.h */
       bzero(&qp_init_attr, sizeof qp_init_attr);
       qp_init_attr.cap.max_recv_sge = MAX_SGE; /* Maximum SGE per receive posting */
       qp_init_attr.cap.max_recv_wr = MAX_WR; /* Maximum receive posting capacity */
       qp_init_attr.cap.max_send_sge = MAX_SGE; /* Maximum SGE per send posting */
       qp_init_attr.cap.max_send_wr = MAX_WR; /* Maximum send posting capacity */
       qp_init_attr.qp_type = IBV_QPT_RC; /* QP type, RC = Reliable connection */
       /* We use same completion queue, but one can use different queues */
       qp_init_attr.recv_cq = client_cq; /* Where should I notify for receive completion operations */
       qp_init_attr.send_cq = client_cq; /* Where should I notify for send completion operations */
       /*Lets create a QP */
       ret = rdma_create_qp(cm_client_id /* which connection id */,
		       pd /* which protection domain*/,
		       &qp_init_attr /* Initial attributes */);
	if (ret) {
		rdma_error("Failed to create QP, errno: %d \n", -errno);
	       exit(1);
	}
	client_qp = cm_client_id->qp;
	debug("QP created at %p \n", client_qp);
	return 0;
}
static int client_connect_to_server() 
{
	struct rdma_conn_param conn_param;
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
	bzero(&conn_param, sizeof(conn_param)); //rdma_conn_param 结构体变量 conn_param，并使用 bzero() 函数将其初始化为零。这个结构体用于设置连接参数，包括 initiator_depth、responder_resources 和 retry_count。
	conn_param.initiator_depth = 15;  //表示连接的发起者（客户端）可以同时发送的最大并发请求数。在这里，我们将其设置为 3，表示客户端可以同时发送最多 3 个请求
	conn_param.responder_resources = 15; // responder_resources 表示连接的响应者（服务器）可以同时处理的最大并发请求数。同样地，我们将其设置为 3。
	conn_param.retry_count = 60; // if fail, then how many times to retry
	debug("cm_client_id %p \n", cm_client_id);
	ret = rdma_connect(cm_client_id, &conn_param); //使用 rdma_connect() 函数来发起与服务器的连接。该函数接受一个 rdma_cm_id 结构体指针 cm_client_id 和一个 rdma_conn_param 结构体指针 conn_param 作为参数。
	if (ret) {
		rdma_error("Failed to connect to remote host , errno: %d\n", -errno);
		return -errno;
	}
	debug("waiting for cm event: RDMA_CM_EVENT_ESTABLISHED\n");
	ret = process_rdma_cm_event(cm_event_channel,   //process_rdma_cm_event 函数等待并处理 RDMA_CM_EVENT_ESTABLISHED 事件，该事件表示客户端与服务器的连接已建立。
			RDMA_CM_EVENT_ESTABLISHED,
			&cm_event);
	if (ret) {
		printf("Failed to get cm event, ret = %d \n", ret);
	       return ret;
	}
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge cm event, errno: %d\n", 
			       -errno);
		return -errno;
	}
	printf("The client is connected successfully \n");
	return 0;
}
std::vector<ibv_mr *>  client_register_mrs(std::vector<ggml_tensor *> &tensor_src)
{	
	std::vector<ibv_mr *> client_src_mrs(tensor_src.size());
	struct ibv_wc wc[2];
	int ret = -1;
	for(int i=0;i<tensor_src.size();++i){
		if(tensor_src[i]->data==NULL)
        {
            printf("tensor_src[%d]->data==NULL\n",i);
        }
		else
		{
			printf("tensor_src[%d]->data!=NULL\n",i);
		
		}
	client_src_mrs[i] = rdma_buffer_register(pd, //rdma_buffer_register函数来注册一个名为client_src_mr的内存区域。这个内存区域包含了客户端要发送给服务器的数据。注册内存区域时，指定了访问权限，包括本地写入、远程读取和远程写入
			tensor_src[i]->data,
			ggml_nbytes(tensor_src[i]),
			(ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|
			 IBV_ACCESS_REMOTE_READ|
			 IBV_ACCESS_REMOTE_WRITE));
			 
	if(!client_src_mrs[i]){
		rdma_error("Failed to register the first buffer, ret = %d \n", ret);
		exit(1);
	}

	}


	return client_src_mrs;
}
//通过存储tensor_dst的信息的client_dst_mrs和服务端server_metadata_attrs的信息，从服务器读取数据
static int client_operation(std::vector<ibv_mr *> client_dst_mrs,rdma_buffer_attr_vec & server_metadata_attrs,ibv_wr_opcode opcode) 
{	
	int size =client_dst_mrs.size();

    int ret = -1;
    struct ibv_wc wc[size];
    ret = process_work_completion_events(io_completion_channel, //process_work_completion_events函数来等待并处理两个工作完成事件。一个是发送工作请求的完成事件，另一个是接收服务器发送的缓冲区信息的完成事件。如果成功接收到两个工作完成事件，代码会打印服务器发送的缓冲区位置和凭证信息。
			wc, 1);

	if(ret != 1) {
		rdma_error("We failed to get 2 work completions , ret = %d \n",
				ret);
		exit(1);
	}
    debug("Server sent us its buffer location and credentials, showing \n");
	show_rdma_buffer_attrs(&server_metadata_attrs);
	/* Step 1: is to copy the local buffer into the remote buffer. We will 
	 * reuse the previous variables. */
	/* now we fill up SGE */ 
	//将本地缓冲区的地址、长度和本地键（lkey）赋值给client_send_sge结构体，表示发送的数据。


	for(int i=0;i<size;++i)
	{
		client_send_sge.addr = (uint64_t) client_dst_mrs[i]->addr;
		client_send_sge.length = (uint32_t) client_dst_mrs[i]->length;
		client_send_sge.lkey = client_dst_mrs[i]->lkey;
		/* now we link to the send work request */ //初始化client_send_wr结构体，并设置相关参数，如SGE列表、SGE数量、操作码（IBV_WR_RDMA_READ）和发送标志（IBV_SEND_SIGNALED）。
		bzero(&client_send_wr, sizeof(client_send_wr));
		client_send_wr.sg_list = &client_send_sge;
		client_send_wr.num_sge = 1;
		client_send_wr.opcode = opcode;
		client_send_wr.send_flags = IBV_SEND_SIGNALED;
		/* we have to tell server side info for RDMA */ // 设置远程RDMA操作的相关信息，包括远程键和远程地址。
		client_send_wr.wr.rdma.rkey = server_metadata_attrs.stags[i].remote_stag;
		client_send_wr.wr.rdma.remote_addr = server_metadata_attrs.address[i];
		/* Now we post it */
		int ret = ibv_post_send(client_qp,  //函数将发送请求发送到RDMA队列中。
				&client_send_wr,
			&bad_client_send_wr);
		if (ret) {
			rdma_error("Failed to read client dst buffer from the master, errno: %d \n", 
					-errno);
			return -errno;
		}
		if((i+1)%5==0)
		{
			ret = process_work_completion_events(io_completion_channel, 
			wc, 5);
			if(ret != 5) {
				rdma_error("We failed to get 2 work completions , ret = %d \n",
						ret);
				return ret;
			}
		}
	/* Now we prepare a READ using same variables but for destination */ //将目标缓冲区的地址、长度和本地键赋值给client_send_sge结构体，表示接收的数据
	}
	ret = process_work_completion_events(io_completion_channel, 
	wc, size%5);
	if(ret != size%5) {
		rdma_error("We failed to get 2 work completions , ret = %d \n",
				ret);
		return ret;
	}
	debug("Client side READ is complete \n");
	return 0;
}
static int client_recv_buffer(rdma_buffer_attr_vec & server_metadata_attrs)
{
	int ret = -1;

	server_metadata_mr = rdma_buffer_register(pd, //rdma_buffer_register函数来注册一个内存区域，该区域用于存储服务器的元数据。rdma_buffer_register函数接受一些参数，包括一个指向内存区域的指针、内存区域的大小以及访问权限。
			&server_metadata_attrs,
			sizeof(server_metadata_attrs),
			(IBV_ACCESS_LOCAL_WRITE));
	if(!server_metadata_mr){
		rdma_error("Failed to setup the server metadata mr , -ENOMEM\n");
		return -ENOMEM;
	}
	server_recv_sge.addr = (uint64_t) server_metadata_mr->addr;
	server_recv_sge.length = (uint32_t) server_metadata_mr->length;
	server_recv_sge.lkey = (uint32_t) server_metadata_mr->lkey;
	/* now we link it to the request */
	bzero(&server_recv_wr, sizeof(server_recv_wr)); //bzero函数将server_recv_wr结构体清零，并将server_recv_sge结构体的地址赋值给server_recv_wr的sg_list成员，将1赋值给server_recv_wr的num_sge成员。这些操作将接收缓冲区的属性与请求相关联。
	server_recv_wr.sg_list = &server_recv_sge;
	server_recv_wr.num_sge = 1;
	ret = ibv_post_recv(client_qp /* which QP */, //代码调用ibv_post_recv函数来提交接收工作请求。该函数接受一些参数，包括一个指向客户端QP（Queue Pair）的指针、一个指向接收工作请求的指针以及一个指向错误工作请求的指针。如果提交成功，函数将返回0，否则返回一个非零值
		      &server_recv_wr /* receive work request*/,
		      &bad_server_recv_wr /* error WRs */);
	if (ret) {
		rdma_error("Failed to pre-post the receive buffer, errno: %d \n", ret);
		return ret;
	}
	debug("Receive buffer pre-posting is successful \n");
	return 0;
}


static int client_disconnect_and_clean_LLM_vec_api(std::vector<ibv_mr *> &client_dst_mrs,std::vector<ibv_mr *> &client_src_mrs) 
{	
	size_t size =client_dst_mrs.size();
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
	/* active disconnect from the client side
{
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
	/* active disconnect from the client side */
	ret = rdma_disconnect(cm_client_id);
	if (ret) {
		rdma_error("Failed to disconnect, errno: %d \n", -errno);
		//continuing anyways
	}
	ret = process_rdma_cm_event(cm_event_channel, 
			RDMA_CM_EVENT_DISCONNECTED,
			&cm_event);
	if (ret) {
		rdma_error("Failed to get RDMA_CM_EVENT_DISCONNECTED event, ret = %d\n",
				ret);
		//continuing anyways 
	}
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge cm event, errno: %d\n", 
			       -errno);
		//continuing anyways
	}
	/* Destroy QP */
	rdma_destroy_qp(cm_client_id);
	/* Destroy client cm id */
	ret = rdma_destroy_id(cm_client_id);
	if (ret) {
		rdma_error("Failed to destroy client id cleanly, %d \n", -errno);
		// we continue anyways;
	}
	/* Destroy CQ */
	ret = ibv_destroy_cq(client_cq);
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
	for(int i=0;i<size;++i)
	{
		rdma_buffer_deregister(client_src_mrs[i]);
		rdma_buffer_deregister(client_dst_mrs[i]);
		
	}
	rdma_buffer_deregister(server_metadata_mr);
	rdma_buffer_deregister(client_metadata_mr);	

	/* We free the buffers */
	// free(tensor_src->data);
	// free(tensor_dst->data);
	/* Destroy protection domain */
	ret = ibv_dealloc_pd(pd);
	if (ret) {
		rdma_error("Failed to destroy client protection domain cleanly, %d \n", -errno);
		// we continue anyways;
	}
	rdma_destroy_event_channel(cm_event_channel);
	printf("Client resource clean up is complete \n");
	return 0;
}



int main(int argc, char **argv) {
	// struct sockaddr_in server_sockaddr;
	int ret, option;
	char * ip;
	char * port = NULL ;
	char * src = NULL;
	char * dst = NULL;
	// bzero(&server_sockaddr, sizeof server_sockaddr);
	// server_sockaddr.sin_family = AF_INET;
	// server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	/* buffers are NULL */
	/* Parse Command Line Arguments */
	while ((option = getopt(argc, argv, "s:a:p:")) != -1) {
		switch (option) {
			case 's':
				printf("Passed string is : %s , with count %u \n", 
						optarg, 
						(unsigned int) strlen(optarg));
				src =(char *) std::calloc(strlen(optarg) , 1);
				if (!src) {
					rdma_error("Failed to allocate memory : -ENOMEM\n");
					return -ENOMEM;
				}
				/* Copy the passes arguments */
				strncpy(src, optarg, strlen(optarg));
				dst = (char *) std::calloc(strlen(optarg), 1);
				if (!dst) {
					rdma_error("Failed to allocate destination memory, -ENOMEM\n");
					free(src);
					return -ENOMEM;
				}
				break;
			case 'a':
				/* remember, this overwrites the port info */
				ip = optarg;
				printf("Passed IP is : %s \n", ip);

				break;
			case 'p':
				/* passed port to listen on */
				port = optarg;
				printf("Passed port is : %s \n", port);
				break;
			default:
				usage();
				break;
			}
		}
	struct sockaddr_in server_sockaddr = get_server_sockaddr(ip, port);
	func_get_ctx();

	//======test_for_ggml_tensor=========//

	//==================================//

	//======test_for_vector_ggml_tensor=========//
	std::vector<struct ibv_mr *> client_src_mrs;
	std::vector<struct ibv_mr *> client_dst_mrs;
	std::vector<struct ggml_tensor *> tensor_srcs;
	std::vector<struct ggml_tensor *> tensor_dsts;
	std::vector<struct ggml_tensor *> tensor_result;
	static struct rdma_buffer_attr_vec server_metadata_attrs;
	static struct rdma_buffer_attr_vec client_metadata_attrs;
	size_t size =10;
	tensor_dsts.resize(size);	
	tensor_srcs.resize(size);
	client_src_mrs.resize(size);
	client_dst_mrs.resize(size);
	// client_metadata_attrs.address.resize(total);
	// client_metadata_attrs.length.resize(total);
	// client_metadata_attrs.stags.resize(total);
	// client_metadata_attrs.size = total;
	for(int i=0;i<size;++i)
	{	
		tensor_srcs[i] = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4);
		tensor_dsts[i] = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4);
		ggml_set_f32(tensor_srcs[i],1.5f);
		printf_value(tensor_srcs[i]);

	}

	if (!server_sockaddr.sin_port) {
	  /* no port provided, use the default port */
	  server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);
	  }

	client_prepare_connection_api(&server_sockaddr);
	printf("start client_pre_post_recv_buffer_LLM_vec_api\n");
	ret = client_recv_buffer(server_metadata_attrs); 
	if (ret) { 
		rdma_error("Failed to setup client connection , ret = %d \n", ret);
		return ret;
	}
	printf("start client_connect_to_server\n");

	ret = client_connect_to_server();
	if (ret) { 
		rdma_error("Failed to setup client connection , ret = %d \n", ret);
		return ret;
	}
		printf("start client_xchange_metadata_with_server_LLM_vec_api\n");

	client_src_mrs= client_register_mrs(tensor_dsts);
	printf("start client_remote_memory_ops_LLM_vec_write_api\n");

	// client_remote_memory_ops_LLM_vec_write_api(client_src_mrs);
	// printf("start client_remote_memory_ops_LLM_vec_register_read_api\n");

	// client_dst_mrs=client_remote_memory_ops_LLM_vec_register_read_api(tensor_dsts);
	ret = client_operation(client_src_mrs,server_metadata_attrs,IBV_WR_RDMA_READ);


	if (ret) {
		rdma_error("Failed to finish remote memory ops, ret = %d \n", ret);
		return ret;
	}
	if (check_src_dst_LLM_vec_api(tensor_srcs,tensor_dsts)) {
		rdma_error("src and dst buffers do not match \n");
	} else {
		printf("...\nSUCCESS, source and destination buffers match \n");
	}
	ret = client_disconnect_and_clean_LLM_vec_api(client_dst_mrs,client_src_mrs);
	if (ret) {
		rdma_error("Failed to cleanly disconnect and clean up resources \n");
	}
	return ret;
}



