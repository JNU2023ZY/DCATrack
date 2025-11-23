import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from lib.models.mixformer2_vit import build_mixformer2_vit_online
from lib.test.evaluation.environment import env_settings  # 需适配你的配置导入

# 1. 加载原PyTorch模型（与跟踪器参数一致）
params = env_settings().get_params_for_tracker("mixformer2_vit_online", "lasot")  # 替换为你的数据集/模型配置
network_fp32 = build_mixformer2_vit_online(params.cfg, train=False)
network_fp32.load_state_dict(
    torch.load(params.checkpoint, map_location="cpu", weights_only=False)["net"],
    strict=True
)
network_fp32.eval()

# 2. 定义输入形状（匹配跟踪器参数）
C = 3  # 图像通道数
template_size = params.template_size  # 如128
search_size = params.search_size      # 如384
# 输入形状：(batch, C, H, W)，online_template支持动态batch（1~online_size）
dummy_template = torch.randn(1, C, template_size, template_size)  # 固定batch=1
dummy_online_template = torch.randn(1, C, template_size, template_size)  # 初始batch=1
dummy_search = torch.randn(1, C, search_size, search_size)  # 固定batch=1

# 3. 导出ONNX（指定三输入+两输出，支持动态维度）
onnx_raw_path = "mixformer_online_raw.onnx"
torch.onnx.export(
    network_fp32,
    args=(dummy_template, dummy_online_template, dummy_search),  # 对应model.forward_test的输入
    f=onnx_raw_path,
    opset_version=14,  # 高版本兼容更多算子
    do_constant_folding=True,
    input_names=["template", "online_template", "search"],  # 输入名称（需与TensorRT绑定一致）
    output_names=["pred_boxes", "pred_scores"],  # 输出名称（对应跟踪器需要的box和score）
    dynamic_axes={
        "online_template": {0: "batch_online"},  # 动态batch（支持1~online_size）
        "pred_boxes": {0: "batch_pred"},
        "pred_scores": {0: "batch_pred"}
    }
)

# 4. 验证ONNX模型有效性
onnx_model = onnx.load(onnx_raw_path)
onnx.checker.check_model(onnx_model)
print(f"ONNX原始模型导出成功：{onnx_raw_path}")

# 5. ONNX INT8动态量化（权重量化，输入动态量化）
onnx_int8_path = "mixformer_online_int8.onnx"
quantize_dynamic(
    model_input=onnx_raw_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8,  # 权重量化为INT8
    op_types_to_quantize=["Conv", "MatMul", "Gemm", "Add"],  # 量化核心算子
    per_channel=False,
    reduce_range=True  # 减小量化范围，提升精度
)
print(f"ONNX INT8量化模型导出成功：{onnx_int8_path}")