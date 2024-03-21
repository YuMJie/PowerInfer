#include<iostream>
#include<cstdio>
#include<cstdlib>
ggml_tensor * create_gpu_rdma(struct ggml_context * aux_ctx,struct ggml_tensor *src, struct ggml_tensor * gpu_bucket) {
#ifdef GGML_USE_CUBLAS
        if (gpu_bucket == NULL) {
            // offload the whole tensor to gpu
            ggml_set_backend(src, GGML_BACKEND_GPU);
            ggml_cuda_transform_tensor(src->data, src);
            return src;
        }

        int64_t row_len = src->ne[0];
        int64_t gpu_rows = gpu_bucket->ne[0];
        GGML_ASSERT(0 < gpu_rows && gpu_rows <= src->ne[1]);

        ggml_set_no_alloc(aux_ctx, true);
        ggml_tensor * gpu_dst = ggml_new_tensor_2d(aux_ctx, src->type, row_len, gpu_rows);
        ggml_set_backend(gpu_dst, GGML_BACKEND_GPU);
        ggml_cuda_alloc_tensor(gpu_dst);

        // init two 1d views on host and device
        ggml_tensor * host_mat_row = ggml_new_tensor_1d(aux_ctx, src->type, row_len);
        static ggml_tensor * device_mat_row = ggml_dup_tensor(aux_ctx, host_mat_row);
        ggml_set_backend(device_mat_row, GGML_BACKEND_GPU);
        ggml_cuda_alloc_tensor(device_mat_row);
        *ggml_cuda_get_data_pp(device_mat_row) = *ggml_cuda_get_data_pp(gpu_dst);

        // read raw data and copy to device depending on gpu_idx
        const enum ggml_type type = src->type;
        const int ne0 = src->ne[0];
        const size_t row_data_size = ne0*ggml_type_size(type)/ggml_blck_size(type);
        for (int i = 0; i < gpu_rows; i++) {
            int32_t host_i = ((int32_t *)gpu_bucket->data)[i];
            host_mat_row -> data = (char *)(src -> data) + host_i * row_data_size;
            char ** gpu_data_pp = reinterpret_cast<char **>(ggml_cuda_get_data_pp(device_mat_row));
            // printf("gpu_data_p: %p\n", *gpu_data_pp);
            ggml_cuda_cpy_1d(device_mat_row, host_mat_row);
            *gpu_data_pp = *gpu_data_pp + row_data_size;
        }
        ggml_set_no_alloc(aux_ctx, false);

        return gpu_dst;
#else
        return NULL;
#endif
}

int main() {
    printf("Hello, World!\n");

    ggml_context * ctx = ggml_new_context();
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_FLOAT, 10, 10);

    
    return 0;
}

