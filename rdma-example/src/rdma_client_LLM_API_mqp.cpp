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
#include "rdma_common.h"
//rdma
#define NUM_QPS 5
static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_client_id = NULL;
static struct rdma_cm_id *cm_client_ids[NUM_QPS];
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_comp_channel *io_completion_channels[NUM_QPS];
static struct ibv_cq *client_cq = NULL;
static struct ibv_cq *client_cqs[NUM_QPS];
static struct ibv_qp *client_qp;
static struct ibv_qp *client_qps[NUM_QPS];
/* These are memory buffers related resources */
static struct ibv_mr *client_metadata_mr = NULL, 
		     *client_src_mr = NULL, 
		     *client_dst_mr = NULL, 
		     *server_metadata_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_send_wr client_send_wr, *bad_client_send_wr = NULL;
static struct ibv_recv_wr server_recv_wr, *bad_server_recv_wr = NULL;
static struct ibv_recv_wr server_recv_wr_1, *bad_server_recv_wr_1 = NULL;
static struct ibv_sge client_send_sge, server_recv_sge;
/* Source and Destination buffers, where RDMA operations source and sink */
// static struct rdma_buffer_attr_vec server_metadata_attrs;
static struct rdma_buffer_attr_vec client_metadata_attrs;

static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp_init_attr qp_init_attrs[NUM_QPS];
std::vector<struct ibv_mr *> client_buffer_mrs;


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static int recv(int qp_index=0)
{
	int ret = -1;

	// client_recv_sge.addr = (uint64_t) server_mr->addr;
	// client_recv_sge.length = (uint32_t) server_mr->length;
	// client_recv_sge.lkey = (uint32_t) server_mr->lkey;
	// /* now we link it to the request */
	// bzero(&client_recv_sge, sizeof(client_recv_sge)); //bzero函数将server_recv_wr结构体清零，并将server_recv_sge结构体的地址赋值给server_recv_wr的sg_list成员，将1赋值给server_recv_wr的num_sge成员。这些操作将接收缓冲区的属性与请求相关联。
	// client_recv_wr.sg_list = &client_recv_sge;
	// client_recv_wr.num_sge = 1;
	ret = ibv_post_recv(client_qps[qp_index] /* which QP */, //代码调用ibv_post_recv函数来提交接收工作请求。该函数接受一些参数，包括一个指向客户端QP（Queue Pair）的指针、一个指向接收工作请求的指针以及一个指向错误工作请求的指针。如果提交成功，函数将返回0，否则返回一个非零值
		      &server_recv_wr_1 /* receive work request*/,
		      &bad_server_recv_wr_1 /* error WRs */);
	if (ret) {
		rdma_error("Failed to pre-post the receive buffer, errno: %d \n", ret);
		return ret;
	}
	debug("Receive buffer pre-posting is successful \n");
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

void printf_value(ggml_tensor * tensor,int num=10)
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
                {   if(--num)
                    {
                            printf("%f ",ggml_get_f32_nd(tensor,l,k,j,i));

                    }
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
	for(int i=0;i<NUM_QPS;++i)
	{
	ret = rdma_create_id(cm_event_channel, &cm_client_ids[i],  //函数创建一个RDMA连接标识符
			NULL,
			RDMA_PS_TCP);
	if (ret) {
		rdma_error("Creating cm id failed with errno: %d \n", -errno); 
		exit(1);
	}
	/* Resolve destination and optional source addresses from IP addresses  to
	 * an RDMA address.  If successful, the specified rdma_cm_id will be bound
	 * to a local device. */
	ret = rdma_resolve_addr(cm_client_ids[i], NULL, (struct sockaddr*) s_addr, 2000); //函数将目标地址解析为RDMA地址，并将cm_client_id与本地设备绑定。
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
	ret = rdma_resolve_route(cm_client_ids[i], 2000); //函数解析到目标地址的RDMA路由，以建立连接
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
	if(!i) pd = ibv_alloc_pd(cm_client_ids[0]->verbs);//函数分配一个保护域（Protection Domain），类似于操作系统中的"进程抽象"，所有资源都与特定的保护域相关联
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
	io_completion_channels[i] = ibv_create_comp_channel(cm_client_ids[i]->verbs); //函数创建一个完成通道（Completion Channel），用于发送I/O完成通知
	if (!io_completion_channels[i]) {
		rdma_error("Failed to create IO completion event channel, errno: %d\n",
			       -errno);
	exit(1);
	}
	debug("completion event channel created at : %p \n", io_completion_channels[i]);
	/* Now we create a completion queue (CQ) where actual I/O 
	 * completion metadata is placed. The metadata is packed into a structure 
	 * called struct ibv_wc (wc = work completion). ibv_wc has detailed 
	 * information about the work completion. An I/O request in RDMA world 
	 * is called "work" ;) 
	 */
	client_cqs[i] = ibv_create_cq(cm_client_ids[i]->verbs /* which device*/,  //ibv_create_cq函数创建一个完成队列（Completion Queue），用于存放实际的I/O完成元数据
			CQ_CAPACITY /* maximum capacity*/, 
			NULL /* user context, not used here */,
			io_completion_channels[i] /* which IO completion channel */, 
			0 /* signaling vector, not used here*/);
	if (!client_cqs[i]) {
		rdma_error("Failed to create CQ, errno: %d \n", -errno);
		exit(1);
	}
	debug("CQ created at %p with %d elements \n", client_cqs[i], client_cqs[i]->cqe);
	ret = ibv_req_notify_cq(client_cqs[i], 0); //ibv_req_notify_cq函数请求通知，以便在完成队列中有新的完成操作时得到通知
	if (ret) {
		rdma_error("Failed to request notifications, errno: %d\n", -errno);
		exit(1);
	}

       /* Now the last step, set up the queue pair (send, recv) queues and their capacity.
         * The capacity here is define statically but this can be probed from the 
	 * device. We just use a small number as defined in rdma_common.h */
       bzero(&qp_init_attrs[i], sizeof qp_init_attrs[i]);
       qp_init_attrs[i].cap.max_recv_sge = MAX_SGE; /* Maximum SGE per receive posting */
       qp_init_attrs[i].cap.max_recv_wr = MAX_WR; /* Maximum receive posting capacity */
       qp_init_attrs[i].cap.max_send_sge = MAX_SGE; /* Maximum SGE per send posting */
       qp_init_attrs[i].cap.max_send_wr = MAX_WR; /* Maximum send posting capacity */
       qp_init_attrs[i].qp_type = IBV_QPT_RC; /* QP type, RC = Reliable connection */
       /* We use same completion queue, but one can use different queues */
       qp_init_attrs[i].recv_cq = client_cqs[i]; /* Where should I notify for receive completion operations */
       qp_init_attrs[i].send_cq = client_cqs[i]; /* Where should I notify for send completion operations */
       /*Lets create a QP */
       ret = rdma_create_qp(cm_client_ids[i] /* which connection id */,
		       pd /* which protection domain*/,
		       &qp_init_attrs[i] /* Initial attributes */);
	if (ret) {
		rdma_error("Failed to create QP, errno: %d \n", -errno);
	       exit(1);
	}
	client_qps[i] = cm_client_ids[i]->qp;
	debug("QP created at %p \n", client_qps[i]);
	}
	return 0;
}

static int client_recv_buffer(rdma_buffer_attr_vec & server_metadata_attrs ,int qp_index=0 )
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
	ret = ibv_post_recv(client_qps[qp_index] /* which QP */, //代码调用ibv_post_recv函数来提交接收工作请求。该函数接受一些参数，包括一个指向客户端QP（Queue Pair）的指针、一个指向接收工作请求的指针以及一个指向错误工作请求的指针。如果提交成功，函数将返回0，否则返回一个非零值
		      &server_recv_wr /* receive work request*/,
		      &bad_server_recv_wr /* error WRs */);
	if (ret) {
		rdma_error("Failed to pre-post the receive buffer, errno: %d \n", ret);
		return ret;
	}
	debug("Receive buffer pre-posting is successful \n");
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
	for(int i=0;i<NUM_QPS;++i)
	{
	debug("cm_client_ids[i] %p \n", cm_client_ids[i]);
	ret = rdma_connect(cm_client_ids[i], &conn_param); //使用 rdma_connect() 函数来发起与服务器的连接。该函数接受一个 rdma_cm_id 结构体指针 cm_client_id 和一个 rdma_conn_param 结构体指针 conn_param 作为参数。
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
	}
	return 0;
}
std::vector<ibv_mr *> register_mrs(std::vector<ggml_tensor *> &tensor_src)
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
void wait_for_server_ready(rdma_buffer_attr_vec & server_metadata_attrs ,int qp_index=0)
{
	printf("Waiting for server to be ready \n");
	struct ibv_wc wc;
	int ret = -1;
    ret = process_work_completion_events(io_completion_channels[qp_index], //process_work_completion_events函数来等待并处理两个工作完成事件。一个是发送工作请求的完成事件，另一个是接收服务器发送的缓冲区信息的完成事件。如果成功接收到两个工作完成事件，代码会打印服务器发送的缓冲区位置和凭证信息。
			&wc, 1);

	if(ret != 1) {
		rdma_error("We failed to get 2 work completions , ret = %d \n",
				ret);
		exit(1);
	}
    debug("Server sent us its buffer location and credentials, showing \n");
	show_rdma_buffer_attrs(&server_metadata_attrs);
}

//通过存储tensor_dst的信息的client_dst_mrs和服务端server_metadata_attrs的信息，从服务器读取数据
static int client_operation(std::vector<ibv_mr *> client_dst_mrs,rdma_buffer_attr_vec & server_metadata_attrs,ibv_wr_opcode opcode,int start,int end,int qp_index=0) 
{	

	int size =client_dst_mrs.size();
	struct ibv_wc wc[size];
	int ret = -1;

	/* Step 1: is to copy the local buffer into the remote buffer. We will 
	 * reuse the previous variables. */
	/* now we fill up SGE */ 
	//将本地缓冲区的地址、长度和本地键（lkey）赋值给client_send_sge结构体，表示发送的数据。
	printf("size=%d\n",size);

	for(int i=0;i<size;++i)
	{	printf("i=%d\n",i);
		client_send_sge.addr = (uint64_t) client_dst_mrs[i]->addr;
		client_send_sge.length = (uint32_t) client_dst_mrs[i]->length;
		client_send_sge.lkey = client_dst_mrs[i]->lkey;
		/* now we link to the send work request */ //初始化client_send_wr结构体，并设置相关参数，如SGE列表、SGE数量、操作码（IBV_WR_RDMA_READ）和发送标志（IBV_SEND_SIGNALED）。
		bzero(&client_send_wr, sizeof(client_send_wr));
		client_send_wr.sg_list = &client_send_sge;
		client_send_wr.num_sge = 1;
		client_send_wr.opcode = opcode;
		client_send_wr.send_flags = 0;
		/* we have to tell server side info for RDMA */ // 设置远程RDMA操作的相关信息，包括远程键和远程地址。
		client_send_wr.wr.rdma.rkey = server_metadata_attrs.stags[i].remote_stag;
		client_send_wr.wr.rdma.remote_addr = server_metadata_attrs.address[i];
		/* Now we post it */
		int ret = ibv_post_send(client_qps[qp_index],  //函数将发送请求发送到RDMA队列中。
				&client_send_wr,
			&bad_client_send_wr);
		if (ret) {
			rdma_error("Failed to read client dst buffer from the master, errno: %d \n", 
					-errno);
			return -errno;
		}
		if(client_send_wr.send_flags&&(i+1)%5==0)
		{
			ret = process_work_completion_events(io_completion_channels[qp_index], 
			wc, 5);
			if(ret != 5) {
				rdma_error("We failed to get 2 work completions , ret = %d \n",
						ret);
				return ret;
			}
		}
	/* Now we prepare a READ using same variables but for destination */ //将目标缓冲区的地址、长度和本地键赋值给client_send_sge结构体，表示接收的数据
	}
	printf("waiting for work completions \n");
	if(client_send_wr.send_flags){
			ret = process_work_completion_events(io_completion_channels[qp_index], 
		wc, size%5);
		if(ret != size%5) {
			rdma_error("We failed to get %d work completions , ret = %d  %s\n",
					size%5,ret, strerror(errno));
			return ret;
		}
	}
	debug("Client side %d is complete \n",opcode);
	return 0;
}

ibv_mr *  register_mrs_and_send(std::vector<ggml_tensor*> tensor_dsts,int qp_index=0 )  //该函数用于向连接的客户端发送服务器端缓冲区的元数据。
{
	struct ibv_wc wc; //工作完成（work completion）结构体
	int ret = -1;

		size_t size =  tensor_dsts.size();
		client_buffer_mrs.resize(size);
		for(int i=0;i<size;++i)
		{
			   client_buffer_mrs[i] = rdma_buffer_register(pd /* which protection domain */, 
		       tensor_dsts[i]->data,
			   ggml_nbytes(tensor_dsts[i]) /* what size to allocate */, 
		       (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|
		       IBV_ACCESS_REMOTE_READ|
		       IBV_ACCESS_REMOTE_WRITE) /* access permissions */);
			if(!client_buffer_mrs[i]){
				rdma_error("Server failed to create a buffer \n");
				/* we assume that it is due to out of memory error */
			}
			client_metadata_attrs.address[i] = (uint64_t) client_buffer_mrs[i]->addr;
			client_metadata_attrs.length[i] = (uint32_t) client_buffer_mrs[i]->length;
			client_metadata_attrs.stags[i].local_stag = (uint32_t) client_buffer_mrs[i]->lkey;
			
		}
			client_metadata_attrs.size = size;
       /* This buffer is used to transmit information about the above 
	* buffer to the client. So this contains the metadata about the client 
	* buffer. Hence this is called metadata buffer. Since this is already 
	* on allocated, we just register it. 
        * We need to prepare a send I/O operation that will tell the 
	* client the address of the client buffer. 
	*/
	//代码准备一个发送操作，用于告知客户端服务器端缓冲区的地址。代码将服务器端缓冲区的地址、长度和本地标签信息填充到client_metadata_attr 结构体中
       ibv_mr * client_metadata_mr = rdma_buffer_register(pd /* which protection domain*/,  //调用 rdma_buffer_register() 函数将其注册到保护域中
		       &client_metadata_attrs /* which memory to register */, 
		       sizeof(client_metadata_attrs) /* what is the size of memory */,
		       IBV_ACCESS_LOCAL_WRITE /* what access permission */);
       if(!client_metadata_mr){
	       rdma_error("Server failed to create to hold client metadata \n");
	       /* we assume that this is due to out of memory error */
       }
       /* We need to transmit this buffer. So we create a send request. 
	* A send request consists of multiple SGE elements. In our case, we only
	* have one 
	*/
		//代码创建一个发送请求，并将 client_metadata_attr 结构体的信息填充到 client_send_sge 结构体中。接着，代码将 client_send_sge 结构体与发送请求关联，并设置发送请求的操作码为 IBV_WR_SEND，表示这是一个发送请求。代码还设置发送请求的标志为 IBV_SEND_SIGNALED，表示希望接收到发送完成的通知。
	//    client_recv_buffer(client_metadata_mr);
	   client_send_sge.addr = (uint64_t) &client_metadata_attrs;
       client_send_sge.length = sizeof(client_metadata_attrs);
       client_send_sge.lkey = client_metadata_mr->lkey;
       /* now we link this sge to the send request */
       bzero(&client_send_wr, sizeof(client_send_wr));
       client_send_wr.sg_list = &client_send_sge;
       client_send_wr.num_sge = 1; // only 1 SGE element in the array 
       client_send_wr.opcode = IBV_WR_SEND; // This is a send request 
       client_send_wr.send_flags = IBV_SEND_SIGNALED; // We want to get notification 
       /* This is a fast data path operation. Posting an I/O request */
	   	// sleep(50);
		ret = ibv_post_send(client_qps[qp_index] /* which QP */,   
		&client_send_wr /* Send request that we prepared before */, 
		&bad_client_send_wr /* In case of error, this will contain failed requests */);
		if (ret) {
			rdma_error("Posting of client metdata failed, errno: %d \n",
					-errno);
		}
	   
	   //代码调用 ibv_post_send() 函数将发送请求提交到客户端的队列对列（QP）中，并检查是否提交成功。

       /* We check for completion notification */
       ret = process_work_completion_events(io_completion_channels[qp_index], &wc, 1);
		debug("Local buffer metadata has been sent to the client \n");
		

		printf("wait writer \n");
		// ret = process_work_completion_events(io_completion_channels[0], &wc, 1);
		// sleep(5);
		debug("Local buffer metadata has been sent to the client \n");
		return  client_metadata_mr;

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


int main(int argc, char ** argv)  {
    std::vector<struct ibv_mr *> client_src_mrs;
	std::vector<struct ibv_mr *> client_dst_mrs;
	std::vector<struct ggml_tensor *> tensor_srcs;
	std::vector<struct ggml_tensor *> tensor_dsts;
	std::vector<struct ggml_tensor *> tensor_result;
	static struct rdma_buffer_attr_vec server_metadata_attrs;
	static struct rdma_buffer_attr_vec client_metadata_attrs;
	char * ip = "10.119.46.62";

	char * port = NULL ;


    
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

    ggml_set_f32(m11, 1.0f);
    printf_nb(m11);
    printf_set(m11);
    printf_value(m11);
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

    printf("\n------ Test 1 - Matrix Mult via F32 code\n");
    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);
    printf("m11xm2_1\n");
    tensor_srcs.push_back(m11xm2);
    //将m11xm2发送给server write
    struct ggml_tensor * m11xm2_1 = ggml_mul_mat(ctx, m11xm2, m12);
    tensor_dsts.push_back(m11xm2_1);
    //得到server的输出后继续计算 read
    printf("m11xm2_2\n");

    struct ggml_tensor * m11xm2_2 = ggml_mul_mat(ctx, m11xm2_1, m13);
    
    printf_value(m11xm2_2);
    // ggml_set_backend(m11xm2, GGML_BACKEND_GPU);

    // printf("Creating compute graph\n");
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m11xm2_2);

    printf("n_threads=%i\n", benchmark_params.n_threads);

    TENSOR_DUMP(m11);
    TENSOR_DUMP(m2);

    //RDMA



	struct sockaddr_in server_sockaddr = get_server_sockaddr(ip, port);
	client_prepare_connection_api(&server_sockaddr);
	client_recv_buffer(server_metadata_attrs,0); 
	client_connect_to_server();
    register_mrs_and_send(tensor_dsts,1);
    client_src_mrs= register_mrs(tensor_srcs);
	wait_for_server_ready(server_metadata_attrs,0);

	recv(0);
	// client_operation(client_dst_mrs,server_metadata_attrs,IBV_WR_RDMA_READ,0,2);
	printf("start write\n");
	sleep(5);
	client_operation(client_src_mrs,server_metadata_attrs,IBV_WR_RDMA_WRITE_WITH_IMM,0,2,1);
	ibv_wc wc;
	printf_value(m11xm2_1);
	printf("wait server write\n");
	process_work_completion_events(io_completion_channels[0], &wc, 1);
	printf_value(m11xm2_1);
    std::vector<uint8_t> work_buffer;

    ggml_graph_compute_helper(work_buffer, gf, benchmark_params.n_threads);

    
    printf("n_nodes=%i\n",gf->n_nodes);
    printf("n_leafs=%i\n",gf->n_leafs);
    for(int i=0;i< gf->n_nodes;i++)
    {
        printf("node %i\n",i);
        printf("node->data %p\n",gf->nodes[i]->data);

        TENSOR_DUMP(gf->nodes[i]);
    }
	// sleep(10);

    return 0;
}
