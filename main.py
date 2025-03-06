from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast, AlignProcessor, AlignModel
import torch
from rapidfuzz import process, utils
import json
import re
import os
from PIL import Image
from typing import List, Dict

app = FastAPI()

# Настройка CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Настройка статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response
 
# Конфигурация моделей
MODEL_NAME_RU = "/app/model"       # Абсолютный путь
MODEL_NAME_EN = "/app/model_en"    # Абсолютный путь
MAX_LENGTH = 128
IMAGE_DIR = "/app/static/photo"
ALIGN_MODEL_PATH = "/app/align-base"

# Инициализация моделей обработки текста
tokenizer_ru = T5TokenizerFast.from_pretrained(MODEL_NAME_RU)
model_ru = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_RU)

generator_ru = pipeline(
    task='text2text-generation',
    model=model_ru,
    tokenizer=tokenizer_ru,
    device=0 if torch.cuda.is_available() else -1,
    max_length=MAX_LENGTH,
    temperature=0.7,
    num_beams=2,
    do_sample=False,
    early_stopping=True,
    num_return_sequences=1,
    top_p=0.95,
    repetition_penalty=1.2
)

tokenizer_en = T5TokenizerFast.from_pretrained(MODEL_NAME_EN)
model_en = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_EN)

generator_en = pipeline(
    task='text2text-generation',
    model=model_en,
    tokenizer=tokenizer_en,
    device=0 if torch.cuda.is_available() else -1,
    max_length=MAX_LENGTH,
    temperature=0.7,
    num_beams=2,
    do_sample=False,
    early_stopping=True,
    num_return_sequences=1,
    top_p=0.95,
    repetition_penalty=1.2
)

# Инициализация Align модели
processor = AlignProcessor.from_pretrained(ALIGN_MODEL_PATH)
align_model = AlignModel.from_pretrained(ALIGN_MODEL_PATH)

class QueryRequest(BaseModel):
    query: str

# Общие утилиты
def normalize_number(num_str: str) -> str:
    s = num_str.lower().strip()
    s = re.sub(r'[тг₸тенге$₽]', '', s)
    s = re.sub(r'\s+', '', s)
    if 'тыс' in s or 'тысяч' in s:
        s = re.sub(r'(тыс|тысяч)', '', s)
        try:
            return str(int(float(s) * 1000))
        except:
            pass
    if 'к' in s:
        s = re.sub(r'к', '', s)
        try:
            return str(int(float(s) * 1000))
        except:
            pass
    s = re.sub(r'\D', '', s)
    return s if s else '0'

# Русская модель
REFERENCE_FILTERS_RU = {
    'action_types': ['аренда', 'покупка', 'продажа', 'продать'],
    'realty_types': ['вилла', 'дом', 'квартира', 'коттедж', 'хостел'],
    'town': ['Алматы', 'Астана', 'Уральск', 'Караганда', "Шымкент", 'Павлодар'],
    'interior_describing': ['без мебели', 'косметический ремонт', 'без косметического ремонта']
}

def extract_rooms_ru(room_str) -> int:
    try:
        room_str = str(room_str).lower().strip()
        match = re.search(r'\d+', room_str)
        if match:
            return int(match.group())
        if any(word in room_str for word in ["однуш", "однокомнат", "1-комнат", "1к"]):
            return 1
        if any(word in room_str for word in ["двуш", "двух", "2-х", "2к", "2х"]):
            return 2
        if any(word in room_str for word in ["трех", "трёх", "3-х", "3к", "3х"]):
            return 3
        return 0
    except Exception as e:
        print(f"Ошибка извлечения комнат: {e}")
        return 0

def parse_model_output_ru(raw_output: str) -> dict:
    parsed = {}
    parts = raw_output.split(';')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip().lower()
            value = value.strip().rstrip(',').replace("_", " ").replace(",", "").strip()
            
            if key == "area":
                parsed["area"] = value
            elif key == "price":
                parsed["price"] = normalize_number(value)
            elif key == "rooms":
                parsed["rooms"] = extract_rooms_ru(value)
            else:
                parsed[key] = value
    return parsed

# Английская модель
REFERENCE_FILTERS_EN = {
    "town": ["Almaty", "Astana", "Shymkent", "Aktau", "Atyrau", "Ust-Kamenogorsk", 
            "Petropavl", "Karaganda", "Aktobe", "Oral", "Kostanay", "Pavlodar", 
            "Taraz", "Kyzylorda", "Semey", "Kokshetau", "Temirtau", "Uralsk"],
    
    "district": ["Medeu", "downtown", "outskirts", "city center", "Bostandyk", 
                "Alatau", "Almaly", "Auezov", "Zhetysu", "Nauryzbay", "Turksib"],
    
    "realty_types": ["studio flat", "villa", "apartment", "flat", "room", 
                    "house", "cottage", "land plot", "garage", "hostel",
                    "hotel", "motel", "guest house", "apart-hotel"],

    "action_types": ["rent", "short-term rent", "long-term rent", "buy", "sell"],
    
    "is_by_homeowner": ["owner", "realtor"],
    
    "photo": ["with photo", "without photo"],
    
    "comfort": ["pets allowed", "free wifi", "soundproofing", "separate bedroom", 
              "charging station", "robot vacuum", "home theater", "projector",
              "mountain view", "smart lock", "smart TV", "high-speed internet"],
    
    'entertainment': ['swimming pool', 'mini bar', 'jacuzzi',
                      'LED lighting', 'game console', 'board games'],
    
    "climate_control": ["air conditioning", "fan", "heater"],
    
    'apart_features': ['balcony', 'unfurnished', 'cosmetic repairs', 'furnished'],
    
    "parc": ["free parking", "underground parking", "paid parking"],
    
    "location_features": ["quiet area", "supermarket", "downtown", "yard view", "city view", "park view", "waterfront view",
                          "skyscraper view", "river view", "sea view", "school", "park"],
    
    "inter_work": ["workspace"],

    "kitchen": ["kitchen", "gas stove", "dining set", "dining area", "electric stove", "drinking water", "refrigerator", "dishes", "sweets",
                "coffee machine", "microwave", "walk-in pantry"],
    
    "photo": ["with photo", "without photo"],

    'family': ['car-sharing', 'baby crib', 'family'],

    'apart_security': [
            'gated community', '24/7 security', 'guarded entrance',
            'CCTV cameras', 'elevator', 'smart lock',
            'video intercom', 'security'],
    
    'bedroom_bath': [
            'shower', 'laundry', 'hygiene products',
            'iron', 'washing machine'],
    
    'nearby': [
            'public transport', 'restaurant', 'coffee shop', 'cafe',
            'metro station', 'bus stop', 'airport', 'hospital',
            'pharmacy', 'clinic', 'sports complex', 'cinema',
            'shopping mall', 'gym', 'spa center', 'car rental',
            'bicycle parking', 'playground', 'beauty salon',
            'store', 'sports ground'],
    
    'international_student': ['international student'],
    
    'expat_friendly': [
            'expat-friendly', 'english-speaking landlord',
            'international community', 'embassy', 'visa support',
            'foreigner registration assistance', 'international school',
            'business center', 'diplomatic district']
}

def extract_rooms_en(text):
    match = re.search(r'(?i)\b(\d+)\s*-?\s*(?:bedrooms?|rooms?)\b', text)
    if not match:
        match = re.search(r'^\D*(\d+)', text)
    return int(match.group(1)) if match else None

def extract_area_en(text):
    match = re.search(r'(\d+\.?\d*)\s*(?:sq\.? ?m|square meters?)', text, re.IGNORECASE)
    if not match:
        match = re.search(r'^(\d+\.?\d*)\b', text)
    return float(match.group(1)) if match else None

def extract_price_en(text):
    cleaned = re.sub(r'[^\d.,]', '', text.replace(',', '.'))
    match = re.search(r'(\d+\.?\d*?)(?:\.\d+)?$', cleaned)
    if match:
        try:
            return int(float(match.group(1)))
        except (ValueError, TypeError):
            return None
    return None

def normalize_text_en(text):
    text = utils.default_process(text)
    text = (
        text.replace("-", " ")
        .replace("_", " ")
        .replace("per day", "daily")
        .replace("long term", "long-term")
        .replace("short term", "short-term")
        .replace("ac", "air conditioning")
        .replace("parking", "parc")
        .replace("with ", "")
        .replace("secondary housing", "secondary")
        .replace("new building", "new")
        .replace("under construction", "construction")
        .strip()
    )
    for word in ["allowed", "friendly", "rental", "photo", "space"]:
        text = text.replace(word, "")
    return text.strip()

def fuzzy_match_en(query, choices, threshold=80):
    norm_query = normalize_text_en(query)
    result = process.extractOne(
        norm_query,
        (normalize_text_en(c) for c in choices),
        score_cutoff=threshold
    )
    return choices[result[2]] if result else None

def map_filters_en(parsed_data):
    mapped = {}
    
    # Обработка числовых полей
    if 'rooms' in parsed_data:
        mapped['rooms'] = extract_rooms_en(str(parsed_data['rooms']))
    
    if 'area' in parsed_data:
        mapped['area'] = extract_area_en(str(parsed_data['area']))
    
    if 'price' in parsed_data:
        price_data = parsed_data['price']
        if isinstance(price_data, list):
            for item in price_data:
                parsed_price = extract_price_en(str(item))
                if parsed_price is not None:
                    mapped['price'] = parsed_price
                    break
        else:
            parsed_price = extract_price_en(str(price_data))
            if parsed_price is not None:
                mapped['price'] = parsed_price

    # Обработка категорий
    main_fields = {
        'town': 80, 'district': 75, 'street': 85,
        'realty_types': 70, 'realty_types2': 70,
        'zhk': 85, 'zastroi': 85, 'action_types': 65,
        'is_by_homeowner': 75, 'photo': 75
    }
    
    for field, threshold in main_fields.items():
        if value := parsed_data.get(field):
            if match := fuzzy_match_en(value, REFERENCE_FILTERS_EN.get(field, []), threshold):
                mapped[field] = match

    # Множественные выборы
    multi_fields = {
        'comfort': 65, 'entertainment': 65, 'apart_features': 70,
        'apart_security': 70, 'inter_work': 75, 'kitchen': 70,
        'location_features': 65, 'parc': 75, 'climate_control': 70,
        'bedroom_bath': 65, 'nearby': 65, 'international_student': 75,
        'expat_friendly': 65, 'family': 70
    }
    
    for field, threshold in multi_fields.items():
        if values := parsed_data.get(field):
            matched = []
            for item in values:
                for part in str(item).replace(' and ', ', ').split(','):
                    part = part.strip()
                    if match := fuzzy_match_en(part, REFERENCE_FILTERS_EN.get(field, []), threshold):
                        if match not in matched:
                            matched.append(match)
            if matched:
                mapped[field] = matched

    return mapped

def process_en_query(query: str):
    result = generator_en(f"Extract tags in JSON format: {query}")
    generated = result[0]['generated_text']
    try:
        parsed_data = json.loads(generated.replace("'", "\""))
    except json.JSONDecodeError:
        parsed_data = {}
        pairs = generated.split(';')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                parsed_data[key.strip()] = value.strip()
    mapped_result = map_filters_en(parsed_data)
    return parsed_data, mapped_result

def get_all_images() -> List[Dict]:
    image_data = []
    for ad_dir in os.listdir(IMAGE_DIR):
        if not ad_dir.isdigit() or int(ad_dir) not in range(1, 31):
            continue
            
        ad_path = os.path.join(IMAGE_DIR, ad_dir)
        if os.path.isdir(ad_path):
            for file in os.listdir(ad_path):
                if file.lower().endswith('.jpg'):
                    image_data.append({
                        "ad_id": ad_dir,
                        "path": os.path.join(ad_path, file),
                        "filename": file
                    })
    return image_data

# Обработчики API
@app.post("/parse")
def parse_ru(req: QueryRequest):
    try:
        result = generator_ru(f"Extract tags in JSON format: {req.query}")
        generated = result[0]['generated_text']
        
        try:
            parsed_data = json.loads(generated.replace("'", "\""))
        except:
            parsed_data = parse_model_output_ru(generated)
        
        # Маппинг для русской версии
        mapped_result = {
            "action_types": fuzzy_match_en(parsed_data.get('action_types', ''), REFERENCE_FILTERS_RU['action_types']),
            "realty_types": fuzzy_match_en(parsed_data.get('realty_types', ''), REFERENCE_FILTERS_RU['realty_types']),
            "town": fuzzy_match_en(parsed_data.get('town', ''), REFERENCE_FILTERS_RU['town']),
            "price": normalize_number(parsed_data.get('price', '')),
            "rooms": extract_rooms_ru(parsed_data.get('rooms', '')),
            "area": parsed_data.get('area', '')
        }
        
        return {"parsed_data": parsed_data, "mapped_result": mapped_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse_en")
def parse_en(req: QueryRequest):
    try:
        parsed_data, mapped_result = process_en_query(req.query)
        return {"parsed_data": parsed_data, "mapped_result": mapped_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import logging
logger = logging.getLogger(__name__)
        
@app.post(
    "/search_images",
    tags=["Image Search"],
    summary="Search images by description",
    response_description="Sorted list of images with scores"
)
def search_images(req: QueryRequest):
    try:
        logger.info("Start processing request...")
        # Парсинг запроса
        parsed_data, mapped_result = process_en_query(req.query)
        
        # Формирование текстового запроса
        query_parts = []
        if realty_type := mapped_result.get('realty_types'):
            query_parts.append(realty_type)
            
        interior_desc = (
            parsed_data.get('interior_describing') or
            parsed_data.get('interior') or
            parsed_data.get('description') or
            parsed_data.get('details')
        )
        if interior_desc:
            query_parts.append(interior_desc)
        
        if not query_parts:
            return {"results": []}
        
        text_query = ", ".join(query_parts)
        
        # Загрузка изображений
        all_images = get_all_images()
        if not all_images:
            return {"results": []}
        
        # Подготовка данных
        images = []
        image_meta = []
        for img in all_images:
            try:
                pil_image = Image.open(img["path"])
                images.append(pil_image)
                image_meta.append(img)
            except Exception as e:
                print(f"Error loading {img['path']}: {str(e)}")
        
        if not images:
            return {"results": []}
        
        # Обработка через модель
        inputs = processor(
            images=images,
            text=[text_query],
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = align_model(**inputs)
        
        # Обработка результатов
        logits = outputs.logits_per_image.squeeze().tolist()
        results = []
        
        for meta, score in zip(image_meta, logits):
            results.append({
                "ad_id": meta["ad_id"],
                "image_url": f"/static/photo/{meta['ad_id']}/{meta['filename']}",
                "score": round(score, 4)
            })
        
        # Сортировка по убыванию score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return {"results": sorted_results}
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))