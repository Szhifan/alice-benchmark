# 测试 encode_with_fields 函数
from transformers import AutoTokenizer
from src.data_prep_alice import encode_with_fields
def test_encode_with_fields():
    # 创建测试数据
    example = {
        "answer": "The capital of France is Paris. It is located in the northern part of the country.",
        "question": "What is the capital of France?",
        "rubric": "Answer must correctly identify Paris as the capital and provide additional context.",
        "sample_solution": "Paris is the capital city of France.",
        "level": 2,
        "sid": "test_001"
    }
    
    # 获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    print("=== 测试 1: 默认参数 (natural_lang format) ===")
    result1 = encode_with_fields(example.copy(), tokenizer, fields=["question","answer", "rubric"], add_instruction=False)
    decoded_text1 = tokenizer.decode(result1["input_ids"])
    print(f"编码文本: {decoded_text1}")
    print(f"input_ids长度: {len(result1['input_ids'])}")
    
    print("\n=== 测试 2: structured format ===")
    result2 = encode_with_fields(example.copy(), tokenizer, format="structured")
    decoded_text2 = tokenizer.decode(result2["input_ids"])
    print(f"编码文本: {decoded_text2}")
    
    print("\n=== 测试 3: 添加指令 ===")
    result3 = encode_with_fields(example.copy(), tokenizer, add_instruction=True)
    decoded_text3 = tokenizer.decode(result3["input_ids"])
    print(f"编码文本: {decoded_text3}")
    
    print("\n=== 测试 4: 自定义字段 ===")
    result4 = encode_with_fields(example.copy(), tokenizer, fields=["question", "answer", "sample_solution"])
    decoded_text4 = tokenizer.decode(result4["input_ids"])
    print(f"编码文本: {decoded_text4}")
    
    print("\n=== 测试 5: structured format + 指令 ===")
    result5 = encode_with_fields(example.copy(), tokenizer, 
                                fields=["answer", "rubric"], 
                                add_instruction=True, 
                                format="structured")
    decoded_text5 = tokenizer.decode(result5["input_ids"])
    print(f"编码文本: {decoded_text5}")
    
    # 验证返回的字段
    print(f"\n=== 返回的字段 ===")
    print(f"字段: {list(result1.keys())}")

if __name__ == "__main__":
    test_encode_with_fields()