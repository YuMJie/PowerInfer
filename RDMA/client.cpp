#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <infiniband/verbs.h>

#define BUFFER_SIZE 1024
#define RDMA_PORT 1

struct rdma_context {
    struct ibv_context* context;
    struct ibv_pd* pd;
    struct ibv_mr* mr;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_port_attr port_attr;
    char* buffer;
};

void create_rdma_context(struct rdma_context* ctx, struct ibv_device* ib_dev) {
    ctx->context = ibv_open_device(ib_dev);
    ctx->pd = ibv_alloc_pd(ctx->context);
    ctx->mr = ibv_reg_mr(ctx->pd, ctx->buffer, BUFFER_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    ctx->cq = ibv_create_cq(ctx->context, 1, NULL, NULL, 0);
    ctx->qp = ibv_create_qp(ctx->pd, NULL);

    ibv_query_port(ctx->context, RDMA_PORT, &ctx->port_attr);

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = RDMA_PORT;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ;

    ibv_modify_qp(ctx->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

void destroy_rdma_context(struct rdma_context* ctx) {
    ibv_destroy_qp(ctx->qp);
    ibv_destroy_cq(ctx->cq);
    ibv_dereg_mr(ctx->mr);
    ibv_dealloc_pd(ctx->pd);
    ibv_close_device(ctx->context);
}

int main() {
    struct rdma_context ctx;
    struct ibv_device** dev_list;
    int num_devices;

    dev_list = ibv_get_device_list(&num_devices);
    if (num_devices == 0) {
        std::cerr << "No RDMA devices found." << std::endl;
        return 1;
    }

    ctx.buffer = new char[BUFFER_SIZE];
    memset(ctx.buffer, 0, BUFFER_SIZE);

    create_rdma_context(&ctx, dev_list[0]);

    // Perform RDMA read operation here

    destroy_rdma_context(&ctx);
    delete[] ctx.buffer;

    ibv_free_device_list(dev_list);

    return 0;
}