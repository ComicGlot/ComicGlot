from transformers import MarianTokenizer, MarianMTModel

model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

input_text = "<등장인물 요약 일람> 이름 : 신유승 전용 특성 :비스트 테이머 (희귀), 반성적 살해자 (일반) 전용 스킬 :[길들이기 Lv.?]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translated_text)