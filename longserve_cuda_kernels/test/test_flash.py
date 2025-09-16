import torch
import longserve_cuda_kernels
import common

def flash_decoding_stage1(input_tensors, parameters):
    mid_out, mid_out_logsumexp, q, k, v, req_to_tokens, b_req_idx, b_seqlen = input_tensors
    # 提取所有必要参数
    k_stride_0 = parameters['k_stride_0']
    k_stride_1 = parameters['k_stride_1']
    v_stride_0 = parameters['v_stride_0']
    v_stride_1 = parameters['v_stride_1']
    sm_scale = parameters['sm_scale']
    batch_size = parameters['batch_size']
    num_q_heads = parameters['num_q_heads']
    num_kv_heads = parameters['num_kv_heads']
    num_blocks = parameters['num_blocks']
    head_dim = parameters['head_dim']
    max_seq_len = parameters['max_seq_len']
    seq_block_size = parameters['seq_block_size']
    max_len_in_batch = parameters['max_len_in_batch']
    block_seq = 8
    # 调用CUDA内核，传递所有参数并保持正确顺序
    longserve_cuda_kernels.flash_decoding_stage1(
        mid_out, 
        mid_out_logsumexp,
        q,
        k,
        v,
        req_to_tokens,
        b_req_idx,
        b_seqlen,
        max_len_in_batch,
        block_seq
    )



if __name__ == "__main__":
    common.set_up_env()

    # 参数设置示例
    batch_size = 105
    num_q_heads = 12
    head_dim = 128  
    num_kv_heads = 12
    num_blocks = 5
    max_seq_len = 512

    seq_block_size = 16  
    max_len_in_batch = max_seq_len  

    # 模拟数据，将类型转换为 float16 (half precision)
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.float16).cuda()
    k = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.float16).cuda()
    v = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.float16).cuda()
    req_to_tokens = torch.randint(0, max_seq_len, (batch_size, max_seq_len), dtype=torch.int32).cuda()
    b_req_idx = torch.randint(0, batch_size, (batch_size,), dtype=torch.int32).cuda()
    b_seqlen = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32).cuda()
    
    mid_out = torch.zeros(num_blocks, batch_size, num_q_heads, head_dim, dtype=torch.float32).cuda()
    mid_out_logsumexp = torch.zeros(num_blocks, batch_size, num_q_heads, dtype=torch.float32).cuda()

    # 创建参数字典，包含所有必要参数
    parameters = {
        'k_stride_0': k.stride(0),
        'k_stride_1': k.stride(1),
        'v_stride_0': v.stride(0),
        'v_stride_1': v.stride(1),
        'sm_scale': 1.0,
        'batch_size': batch_size,
        'num_q_heads': num_q_heads,
        'num_kv_heads': num_kv_heads,
        'num_blocks': num_blocks,
        'head_dim': head_dim,
        'max_seq_len': max_seq_len,
        'seq_block_size': seq_block_size,
        'max_len_in_batch': max_len_in_batch  # 需根据实际数据计算
    }

    # 确保输入张量正确
    input_tensors = (mid_out, mid_out_logsumexp, q, k, v, req_to_tokens, b_req_idx, b_seqlen)
    
    flash_decoding_stage1(input_tensors, parameters)

    #print(mid_out)
    print("CUDA 内核调用成功")