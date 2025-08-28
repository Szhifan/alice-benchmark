import os
import shutil
import torch
from transformers import AutoConfig
from modelling.modelling_snet import AsagSNet



def test_save_and_load_with_peft():
    # 创建临时目录
    from peft import LoraConfig
    save_dir = "temp_model_peft"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # 初始化模型
    config = AutoConfig.from_pretrained("bert-base-uncased")
    print(config)
    config.num_labels = 2
    config.pool_type = "mean"
    config.base_model_name_or_path = "bert-base-uncased"
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        bias='none',
        target_modules="all-linear",
        task_type=None 
    )
    model = AsagSNet(config, lora_config=lora_config)
    model.init_peft()

    # 保存模型
    model.save_pretrained(save_dir)

    # 加载模型
    loaded_model = AsagSNet.from_pretrained(save_dir, lora_config=lora_config)

    # 检查参数是否一致
    for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(param1, param2), "Parameters do not match!"

    print("Test with PEFT passed!")

if __name__ == "__main__":
    test_save_and_load_with_peft()