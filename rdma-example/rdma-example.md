# RDMA-EXAMPLE

​																						

​		

| 服务器                                                       | 客户端                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 监听端口 ，通过process_rdma_cm_event获得新连接的客户端的标识符 | 创建了一个RDMA事件通道和RDMA连接标识符。函数将目标地址解析为RDMA地址，并将cm_client_id与本地设备绑定。并创建接受队列 |
| 注册用于接收client_metadata_attr的缓冲区client_metadata_mr   | 注册用于接收server_metadata_attr的缓冲区server_metadata_mr,把该缓冲区的信息绑定到server_recv_wr中 |
| 调用ibv_post_recv 接收client_metadata_attr，，指定所接收到的数据存放client_metadata_mr（请求1） | 调用ibv_post_recv函数提交接收请求，指定所接收到的数据存放server_metadata_mr（请求1）(如何确保得到的是远程对应的数据) |
| rdma_accept()函数接受客户端的连接请求，并传入连接参数。等待RDMA_CM_EVENT_ESTABLISHED | 使用 rdma_connect() 函数来发起与服务器的连接。等待RDMA_CM_EVENT_ESTABLISHED |
|                                                              | 为发送数据src注册缓冲区client_src_mr，将缓存区的元数据存储在client_metadata_attr |
| process_work_completion_events函数等待接收服务器发送的缓冲区信息client_metadata_attr的完成事件（请求1）。 | 注册用于发送client_metadata_attr的内存区域client_metadata_mr。调用ibv_post_send函数将发送client_metadata_mr请求提交到客户端的队列。（请求2） |
| 根据client_metadata_attr给注册用于server_buffer_mr的缓冲区，供客户端访问 |                                                              |
| 根据server_buffer_mr把server_metadata_attr进行赋值，注册用于发送server_metadata_attr的缓冲区server_metadata_mr |                                                              |
| 调用ibv_post_send把server_metadata_attr发送给客户端（请求2） |                                                              |
| process_work_completion_events等待发送server_metadata_attr的工作请求的完成（请求2）。 | process_work_completion_events等待发送client_metadata_attr的工作请求的完成（请求2）和接收服务器发送的缓冲区信息server_metadata_mr的完成事件（请求1）。 |
|                                                              | 写操作：把缓冲区client_src_mr的元数据写入发送请求client_send_wr中，并指定写操作以及远程写区域的缓冲区是server_metadata_attr。调用ibv_post_send把client_send_wr发送到RDMA队列中。 |
|                                                              | process_work_completion_events等待发送的完成。此时src的数据已经写在远程缓冲区server_buffer_mr上了 |
|                                                              | 读操作：注册用于接收远程数据的缓冲区client_dst_mr。把缓冲区client_dst_mr的元数据写到client_send_wr，指定读操作以及远程读区域的缓冲区是server_metadata_attr。调用ibv_post_send把client_send_wr发送到RDMA队列中。 |
|                                                              | process_work_completion_events等待发送的完成。此时远程缓冲区client_dst_mr的数据已经写在本地缓冲区client_dst_mr上了 |
| 等待客户端发送中断连接的请求                                 | 断开连接                                                     |

