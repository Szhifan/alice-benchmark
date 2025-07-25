from transformers import AutoTokenizer
from src.data_prep import encode_with_fields

# 创建一个模拟的tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 创建测试样例
example = {
    "answer": "The capital is Paris",
    "question": "What is the capital of France?",
    "sample_solution": "Paris is the capital city of France",
    "rubric": "Answer must correctly identify Paris as the capital",
    "sid": "test_001"
}

print("=== 测试1: 默认参数 (只有answer字段) ===")
result1 = encode_with_fields(example.copy(), tokenizer)
print("编码文本应该是:")
print("answer: The capital is Paris\nrubric: Answer must correctly identify Paris as the capital")
print(f"input_ids长度: {len(result1['input_ids'])}")
print(f"token: {tokenizer.convert_ids_to_tokens(result1['input_ids'])}")
print()

print("=== 测试2: 多个字段 (question + answer) ===")
result2 = encode_with_fields(example.copy(), tokenizer, fields=["question", "answer"])
print("编码文本应该是:")
print("question: What is the capital of France?\nanswer: The capital is Paris\nrubric: Answer must correctly identify Paris as the capital")
print(f"input_ids长度: {len(result2['input_ids'])}")
print(f"token: {tokenizer.convert_ids_to_tokens(result2['input_ids'])}")
print()

print("=== 测试3: 添加指令 ===")
result3 = encode_with_fields(example.copy(), tokenizer, fields=["answer"], add_instruction=True)
print("编码文本应该是:")
print("Determine if rubric is satisfied by the answer:\nanswer: The capital is Paris\nrubric: Answer must correctly identify Paris as the capital")
print(f"input_ids长度: {len(result3['input_ids'])}")
print(f"token: {tokenizer.convert_ids_to_tokens(result3['input_ids'])}")
print()

print("=== 测试4: 结构化格式 ===")
result4 = encode_with_fields(example.copy(), tokenizer, fields=["answer"], format="structured")
print("编码文本应该是:")
print("<answer>The capital is Paris</answer>\n<rubric>Answer must correctly identify Paris as the capital</rubric>")
print(f"input_ids长度: {len(result4['input_ids'])}")
print(f"token: {tokenizer.convert_ids_to_tokens(result4['input_ids'])}")
print()

print("=== 测试5: 结构化格式 + 指令 + 多字段 ===")
result5 = encode_with_fields(example.copy(), tokenizer, 
                           fields=["question", "sample_solution","answer"], 
                           add_instruction=True, 
                           format="structured")
print("编码文本应该是:")
print("Determine if rubric is satisfied by the answer:\n<question>What is the capital of France?</question>\n<sample_solution>Paris is the capital city of France</sample_solution>\n<rubric>Answer must correctly identify Paris as the capital</rubric>")
print(f"input_ids长度: {len(result5['input_ids'])}")
print(f"token: {tokenizer.convert_ids_to_tokens(result5['input_ids'])}")
print()

print("=== 测试6: 错误处理 - 不存在的字段 ===")
try:
    result6 = encode_with_fields(example.copy(), tokenizer, fields=["nonexistent_field"])
except ValueError as e:
    print(f"捕获到期望的错误: {e}")
print()

print("=== 测试7: 检查返回的example是否包含tokenizer输出 ===")
result7 = encode_with_fields(example.copy(), tokenizer, fields=["answer"])
print("返回的example应该包含以下tokenizer字段:")
for key in result7:
    if key in ['input_ids', 'attention_mask', 'token_type_ids']:
        print(f"- {key}: {type(result7[key])}, 长度: {len(result7[key])}")
    else:
        print(f"- {key}: {result7[key]}")