from transformers import T5Tokenizer, T5ForConditionalGeneration, T5TokenizerFast
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)

# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

tokenizer = T5TokenizerFast.from_pretrained('paust/pko-flan-t5-large')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-flan-t5-large').to(device)

input_text = """
OCR1: 〈등장인물〉 이름 : 신유승 전용 특성 : 비스트 테이머 <희귀>, 반성적 살해자 <일반> 전용 스킬 : [길들이기 lv.7]
OCR2: <등장인물 요약 일람> 이름 : 신유승 전용 특성 :비스트 테이머 (희귀), 반성적 살해자 (일반) 전용 스킬 :[길들이기 Lv.?]
동일한 텍스트에 대한 두 개의 OCR 출력이 주어지면 가장 정확한 버전을 재구성합니다.
"""

input_ids = tokenizer(input_text, add_special_tokens=True, return_tensors='pt').input_ids.to(device)
output_ids = model.generate(input_ids=input_ids.cuda(), max_new_tokens=100, num_beams=12)
text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(text)