import tensorly as tl
import numpy as np
from tensorly.decomposition import tucker

def tensortucker(tensor, ranks=None):
    try:
        tensor = tensor.numpy()
        core, factors = tucker(tensor, ranks)
        reconstructed = tl.tucker_to_tensor((core, factors))
        # 计算重构误差
        error = tl.norm(tensor - reconstructed) / tl.norm(tensor)
        # print(f"\n重构相对误差: {error:.6f}")

        return core, error

    except np.linalg.LinAlgError as e:
        if "SVD did not converge" in str(e):
            return tensor, 1.0  # 1.0表示不收敛错误
        else:
            print(f"线性代数错误: {e}")
            return tensor, 1.0

"""
# 示例用法
if __name__ == "__main__":
    # 设置随机种子以便复现结果
    np.random.seed(42)

    # 创建一个 3阶张量 (3x4x5)
    tensor = tl.tensor(np.random.random((2, 100, 200)))
    print("原始张量形状:", tensor.shape)


    core,error = tensortucker(tensor, (2, 64, 128))
    print(core.shape)
    
"""
