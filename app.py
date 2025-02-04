from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import ast
import re
import json
import math
import traceback

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 전역 변수로 style_mapping 정의
style_mapping = {
    '스포츠': ['스포츠', '운동', '애슬레저', '스포츠웨어'],
    '캐주얼': ['캐주얼', '데일리룩', '일상복', '편한'],
    '데일리': ['데일리', '일상', '평상복', '기본'],
    '페미닌': ['페미닌', '여성스러운', '로맨틱', '걸리시'],
    '포멀': ['포멀', '정장', '비즈니스', '격식'],
    '섹시': ['섹시', '글래머', '섹시한', '볼륨'],
    '미니멀': ['미니멀', '심플', '단순한', '깔끔한'],
    '스포티': ['스포티', '스포티브', '액티브', '활동적인'],
    '럭셔리': ['럭셔리', '고급스러운', '명품', '프리미엄'],
    '스트릿': ['스트릿', '길거리', '힙합', '스케이터'],
    '트렌디': ['트렌디', '유행', '최신', '인기'],
    '클래식': ['클래식', '고전적인', '전통적인', '베이직'],
    '빈티지': ['빈티지', '레트로', '구제', '올드스쿨'],
    '보헤미안': ['보헤미안', '보헤미안룩', '자유로운', '히피'],
    '글래머러스': ['글래머러스', '화려한', '돋보이는', '세련된'],
    '프레피': ['프레피', '학생룩', '아카데믹', '교복'],
    '아웃도어': ['아웃도어', '등산복', '야외활동', '캠핑'],
    '유니크': ['유니크', '독특한', '개성있는', '특이한'],
    '컨템포러리': ['컨템포러리', '현대적인', '모던한', '세련된']
}

# mall_domains와 default_image 정의 유지
mall_domains = {
    'carenel': 'https://carenel.com',
    'coconco': 'https://www.coconco.com',
    'dailybain': 'https://dailybain.com',
    'nabiang': 'https://www.nabiang.co.kr',
    'naning9': 'https://www.naning9.com',
    'neriah': 'https://neriah.kr',
    'pink-rocket': 'http://www.pink-rocket.com',
    'vanillashu': 'https://www.vanillashu.co.kr',
    'varzar': 'https://varzar.com',
    'musinsa': 'https://www.musinsa.com',
    '29cm': 'https://www.29cm.co.kr',
    'wconcept': 'https://www.wconcept.co.kr',
    'ssf': 'https://www.ssfshop.com',
    'hfashionmall': 'https://www.hfashionmall.com',
    'thehandsome': 'https://www.thehandsome.com',
    'sivillage': 'https://www.sivillage.com',
    'lfmall': 'https://www.lfmall.co.kr',
    'hmall': 'https://www.hyundaihmall.com',
    # 필요한 쇼핑몰 도메인을 추가하세요
}

default_image = '/static/no-image.png'

# 전역 변수로 데이터프레임 선언
df = None

# 상수 정의 부분의 카테고리 매핑 수정
CATEGORY_MAPPING = {
    'TOP': ['셔츠&블라우스', '반팔티', '니트', '후드/맨투맨', '긴팔티', '트레이닝', '나시'],
    'PANTS': ['롱팬츠', '슬랙스', '레깅스', '숏팬츠', '트레이닝'],
    'DRESS&SKIRT': ['스커트'],
    'OUTER': ['가디건', '베스트', '자켓', '점퍼', '집업'],
    'BEST': ['펌프스', '샌들/뮬', '스니커즈', '플랫/로퍼', '부츠/앵클', '슬링백', '블로퍼/슬리퍼'],
    'BAG': ['크로스백', '미니백', '숄더백', '토드백', '캔버스/백팩', '클러치/파우치'],
    'ACC': ['모자', '귀걸이', '목걸이'],
    'UNDERWEAR': ['브라', '팬티', '보정']
}

# 검색어 매핑 정의
SEARCH_KEYWORD_MAPPING = {
    '모자': {
        'categories': ['모자', '볼캡', '버킷햇'],
        'keywords': ['모자', '볼캡', '버킷햇', '캡'],
        'description_keywords': ['모자', '볼캡', '버킷햇', '캡', '햇', '비니']
    },
    '바지': {
        'categories': ['PANTS', '팬츠', '바지', '슬랙스', '진'],
        'keywords': ['바지', '팬츠', '슬랙스', '진', '청바지', '데님'],
        'description_keywords': ['바지', '팬츠', '슬랙스', '진', '청바지', '데님', '와이드', '스트레이트']
    },
    '치마': ['DRESS&SKIRT > 스커트'],
    '티셔츠': ['TOP > 반팔티', 'TOP > 긴팔티'],
    '자켓': ['OUTER > 자켓'],
    '원피스': ['DRESS&SKIRT'],
    '가방': ['BAG'],
    '신발': ['BEST']
}

# 역방향 카테고리 매핑 생성
REVERSE_CATEGORY_MAPPING = {}
for main_category, sub_categories in CATEGORY_MAPPING.items():
    for sub_category in sub_categories:
        REVERSE_CATEGORY_MAPPING[sub_category] = main_category
        REVERSE_CATEGORY_MAPPING[f"{main_category} > {sub_category}"] = main_category

# 카테고리 매핑 정의
CATEGORY_KEYWORDS = {
    '모자': ['모자', '볼캡', '캡', '베레모', '버킷햇'],
    '상의': ['상의', '티셔츠', '셔츠', '블라우스', '니트', '맨투맨', '후드'],
    '하의': ['하의', '바지', '청바지', '팬츠', '스커트', '레깅스'],
    '원피스': ['원피스', '드레스'],
    '아우터': ['아우터', '자켓', '코트', '패딩', '가디건'],
    '신발': ['신발', '운동화', '구두', '슬리퍼', '샌들'],
    '가방': ['가방', '백팩', '크로스백', '숄더백', '클러치'],
    '악세서리': ['악세서리', '귀걸이', '목걸이', '반지', '팔찌']
}

def load_csv():
    global df
    try:
        df = pd.read_csv('tailor_product_20250121.csv')
        print(f"CSV 파일 로드 완료. 총 {len(df)} 개의 제품이 있습니다.")
        return True
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {str(e)}")
        df = None
        return False

# 앱 시작 시 CSV 파일 로드
if not load_csv():
    print("Warning: CSV 파일 로드 실패")

@app.route('/')
def home():
    return render_template('index.html')

def analyze_query(query):
    try:
        prompt = f"""
사용자의 쇼핑 관련 질문을 분석하여 다음 필터 값들을 JSON 형식으로 추출해주세요.

분석 규칙:
1. 실제 검색하고자 하는 상품 키워드만 'keywords'에 포함
2. 색상은 별도로 추출하여 'color' 필드에 포함
3. 가격 제한은 원 단위로 변환하여 숫자만 반환
4. 스타일 분석 규칙:
   - "~에 어울리는", "~와 잘 어울리는", "~랑 매치되는" 등의 코디네이션 문구가 있을 경우:
     * 카고바지 → 스트릿, 캐주얼, 스포티
     * 정장 → 포멀, 클래식
     * 청바지 → 캐주얼, 데일리
     * 미니스커트 → 페미닌, 섹시
     * 트레이닝복 → 스포츠, 스포티
   - 해당 아이템의 일반적인 스타일 특성을 고려하여 적절한 스타일 키워드 선택

입력: {query}

JSON 형식으로만 응답해주세요:
{{
    "style": "다음 중 하나만 선택 [스포츠, 캐주얼, 데일리, 페미닌, 포멀, 섹시, 미니멀, 스포티, 럭셔리, 스트릿, 트렌디, 클래식, 빈티지, 보헤미안, 글래머러스, 프레피, 아웃도어, 유니크, 컨템포러리]",
    "gender": "명시적인 성별 언급이 있는 경우에만 [여성, 남성, 공용] 중 선택, 없으면 null",
    "age_group": "다음 중 하나만 선택 [10대, 20대, 30대, 40대, 50대 이상], 없으면 null",
    "price_limit": "가격 제한이 있는 경우 원 단위 숫자로 변환, 없으면 null",
    "season": "다음 중 선택 [봄, 여름, 가을, 겨울, 간절기] (여러 개 가능), 없으면 null",
    "keywords": "실제 검색하고자 하는 상품 키워드만 포함 (배열)",
    "color": "색상 언급이 있는 경우 매핑된 색상명, 없으면 null"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant that analyzes user queries and extracts structured filter values. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # 응답에서 JSON 부분만 추출
        response_text = response.choices[0].message.content.strip()
        
        # JSON 형식이 아닌 텍스트 제거
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].strip()
            
        # 입력: 등의 텍스트가 있으면 제거
        if '입력:' in response_text:
            response_text = response_text.split('{', 1)[1]
            response_text = '{' + response_text
            
        try:
            # JSON 문자열을 파이썬 딕셔너리로 변환
            filter_values = json.loads(response_text)
            # category 키 제거
            if 'category' in filter_values:
                del filter_values['category']
            return filter_values
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {response_text}")
            print(f"파싱 오류 상세: {str(e)}")
            return None

    except Exception as e:
        print(f"OpenAI API 오류: {str(e)}")
        return None

def extract_category_from_query(query):
    """사용자 질문에서 카테고리 키워드 추출"""
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                return category
    return None

def get_expanded_keywords(keyword):
    """키워드에 대한 확장 검색어 목록을 반환"""
    if keyword in SEARCH_KEYWORD_MAPPING:
        mapping = SEARCH_KEYWORD_MAPPING[keyword]
        return {
            'categories': mapping.get('categories', []),
            'keywords': mapping.get('keywords', []),
            'description_keywords': mapping.get('description_keywords', [])
        }
    return {
        'categories': [keyword],
        'keywords': [keyword],
        'description_keywords': [keyword]
    }

def convert_price_to_number(price_str):
    """가격 문자열을 숫자로 변환 (예: "30,000원" -> 30000)"""
    try:
        if pd.isna(price_str):
            return float('inf')
        # 쉼표와 '원' 제거 후 숫자로 변환
        price_str = str(price_str)
        return int(price_str.replace(',', '').replace('원', ''))
    except:
        return float('inf')

def get_category_mapping():
    """카테고리 매핑 정의"""
    return {
        # 상의 (TOP)
        '상의': [
            'TOP > 셔츠&블라우스',
            'TOP > 반팔티',
            'TOP > 니트',
            'TOP > 후드/맨투맨',
            'TOP > 긴팔티',
            'TOP > 트레이닝',
            'TOP > 나시'
        ],
        
        # 하의 (PANTS)
        '하의': [
            'PANTS > 롱팬츠',
            'PANTS > 슬랙스',
            'PANTS > 레깅스',
            'PANTS > 숏팬츠',
            'PANTS > 트레이닝'
        ],
        
        # 원피스/스커트
        '원피스/스커트': [
            'DRESS&SKIRT > 스커트'
        ],
        
        # 아우터 (OUTER)
        '아우터': [
            'OUTER > 가디건',
            'OUTER > 베스트',
            'OUTER > 자켓',
            'OUTER > 점퍼',
            'OUTER > 집업'
        ],
        
        # 신발
        '신발': [
            'BEST > 펌프스',
            'BEST > 샌들/뮬',
            'BEST > 스니커즈',
            'BEST > 플랫/로퍼',
            'BEST > 부츠/앵클',
            'BEST > 슬링백',
            'BEST > 블로퍼/슬리퍼'
        ],
        
        # 가방
        '가방': [
            '크로스백',
            '미니백',
            '숄더백',
            '토드백',
            '캔버스/백팩',
            '클러치/파우치'
        ],
        
        # 모자
        '모자': [
            '모자 > 볼캡',
            '모자 > 버킷햇'
        ],
        
        # 악세서리
        '악세서리': [
            '귀걸이',
            '목걸이'
        ],
        
        # 언더웨어
        '언더웨어': [
            '브라',
            '팬티',
            '보정'
        ]
    }

def get_category_from_keywords(keywords):
    """키워드를 기반으로 카테고리 추정"""
    category_keywords = {
        '상의': ['티셔츠', '셔츠', '블라우스', '니트', '맨투맨', '후드', '나시'],
        '하의': ['바지', '팬츠', '슬랙스', '레깅스', '숏팬츠', '트레이닝'],
        '원피스/스커트': ['원피스', '스커트'],
        '아우터': ['자켓', '코트', '가디건', '점퍼', '집업'],
        '신발': ['신발', '운동화', '구두', '샌들', '슬리퍼', '부츠', '스니커즈'],
        '가방': ['가방', '백팩', '크로스백', '숄더백', '클러치'],
        '모자': ['모자', '캡', '버킷햇'],
        '악세서리': ['귀걸이', '목걸이'],
        '언더웨어': ['브라', '팬티', '속옷']
    }
    
    for keyword in keywords:
        for category, words in category_keywords.items():
            if any(word in keyword for word in words):
                return category
    return None

def apply_category_filter(df, category):
    """카테고리 필터 적용"""
    if not category:
        return df
        
    category_mapping = get_category_mapping()
    if category in category_mapping:
        valid_categories = category_mapping[category]
        return df[df['category'].isin(valid_categories)]
    
    return df

def apply_color_filter(df, color):
    """색상 필터 적용"""
    if not color or pd.isna(df['color_option']).all():
        return df
    
    # color_option 컬럼의 값을 파싱
    def check_color(color_options):
        if pd.isna(color_options):
            return False
        # 쉼표로 구분된 색상 옵션을 리스트로 변환
        options = [opt.strip() for opt in str(color_options).split(',')]
        # 괄호 안의 영문 색상명도 체크 (예: "블랙(Black)")
        color_patterns = [
            color,  # 정확한 매칭
            f"{color}(",  # 괄호 시작
            f"({color}",  # 괄호 안
        ]
        return any(any(pattern.lower() in opt.lower() for pattern in color_patterns) for opt in options)
    
    return df[df['color_option'].apply(check_color)]

def apply_season_filter(df, seasons):
    """시즌 필터 적용"""
    if not seasons or pd.isna(df['season']).all():
        return df
    
    # season 컬럼의 값을 파싱
    def check_seasons(season_str):
        if pd.isna(season_str):
            return False
        # 문자열을 리스트로 변환 (예: "['봄', '여름']" -> ['봄', '여름'])
        try:
            season_list = eval(season_str)
            # 검색된 시즌이 상품의 시즌 리스트에 하나라도 포함되어 있으면 True
            return any(season in season_list for season in seasons)
        except:
            return False
    
    return df[df['season'].apply(check_seasons)]

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        page = data.get('page', 1)
        per_page = 16

        print(f"\n=== 검색 시작 ===")
        print(f"입력된 검색어: {query}")
        print(f"페이지: {page}")
        
        # DataFrame 상태 확인
        if df is None:
            print("Error: DataFrame이 로드되지 않았습니다.")
            return jsonify({
                'error': 'CSV 데이터가 로드되지 않았습니다.',
                'results': [],
                'total_count': 0,
                'current_page': page,
                'total_pages': 0,
                'per_page': per_page,
                'filter_details': {}
            })

        print(f"DataFrame 크기: {len(df)} 행")
        
        # 필터 초기화
        filtered_df = df.copy()
        filter_details = {}  # filter_details 초기화 추가
        
        # 필터 값 분석
        filter_values = analyze_query(query)
        print(f"분석된 필터 값: {filter_values}")
        
        if filter_values is None:
            print("Error: 필터 값 분석 실패")
            return jsonify({
                'error': '검색어 분석 중 오류가 발생했습니다.',
                'results': [],
                'total_count': 0,
                'current_page': page,
                'total_pages': 0,
                'per_page': per_page,
                'filter_details': {}
            })

        def apply_filter(df, filter_func, filter_name):
            """필터 적용 함수. 결과가 0이면 이전 결과 반환"""
            previous = df.copy()
            filtered = filter_func(df)
            print(f"필터링 후 데이터 수: {len(filtered)}")
            
            if len(filtered) == 0:
                print(f"⚠️ {filter_name} 필터링 결과가 0입니다. 이전 결과를 유지합니다.")
                return previous
            return filtered
        
        # 스타일 필터링
        if filter_values.get('style'):
            print(f"\n=== 스타일 필터링 ===")
            print(f"검색할 스타일: {filter_values['style']}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            def style_filter(df):
                df['style'] = df['style'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                style_mask = df['style'].apply(
                    lambda x: any(any(s.lower() in style.lower() for style in x) for s in filter_values['style']) 
                    if isinstance(x, list) else False
                )
                return df[style_mask]
            
            filtered_df = apply_filter(filtered_df, style_filter, "스타일")
            filter_details['style'] = {'matched_values': filter_values['style']}
        
        # 연령대 필터링
        if filter_values.get('age_group'):
            print(f"\n=== 연령대 필터링 ===")
            print(f"검색할 연령대: {filter_values['age_group']}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            def age_filter(df):
                df['age_group'] = df['age_group'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                age_mask = df['age_group'].apply(
                    lambda x: filter_values['age_group'] in x if isinstance(x, list) else False
                )
                return df[age_mask]
            
            filtered_df = apply_filter(filtered_df, age_filter, "연령대")
            filter_details['age_group'] = {'matched_values': [filter_values['age_group']]}
        
        # 성별 필터링
        if filter_values.get('gender'):
            print(f"\n=== 성별 필터링 ===")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            def gender_filter(df):
                if filter_values['gender'] == '여성':
                    return df[df['성별(F(여성)/M(남성)/U(공용))'].isin(['F', 'U'])]
                elif filter_values['gender'] == '남성':
                    return df[df['성별(F(여성)/M(남성)/U(공용))'].isin(['M', 'U'])]
                else:  # 공용
                    return df[df['성별(F(여성)/M(남성)/U(공용))'] == 'U']
            
            filtered_df = apply_filter(filtered_df, gender_filter, "성별")
            filter_details['gender'] = {'matched_values': [filter_values['gender']]}
        
        # 카테고리 필터링
        category = get_category_from_keywords(filter_values.get('keywords', []))
        if category:
            print(f"\n=== 카테고리 필터링 ===")
            print(f"검색할 카테고리: {category}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            filtered_df = apply_filter(filtered_df, lambda df: apply_category_filter(df, category), "카테고리")
            if len(filtered_df) > 0:
                filter_details['category'] = {'matched_values': category}
        
        # 색상 필터링
        if filter_values.get('color'):
            print(f"\n=== 색상 필터링 ===")
            print(f"검색할 색상: {filter_values['color']}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            filtered_df = apply_filter(filtered_df, 
                                    lambda df: apply_color_filter(df, filter_values['color']), 
                                    "색상")
            
            if len(filtered_df) > 0:
                filter_details['color'] = {'matched_values': filter_values['color']}
        
        # 시즌 필터링
        if filter_values.get('season'):
            print(f"\n=== 시즌 필터링 ===")
            print(f"검색할 시즌: {filter_values['season']}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            filtered_df = apply_filter(filtered_df, 
                                    lambda df: apply_season_filter(df, filter_values['season']), 
                                    "시즌")
            
            if len(filtered_df) > 0:
                filter_details['season'] = {'matched_values': filter_values['season']}
        
        # 키워드 필터링
        if filter_values.get('keywords'):
            print(f"\n=== 키워드 필터링 ===")
            print(f"검색 키워드: {filter_values['keywords']}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            def keyword_filter(df):
                keyword_conditions = []
                for keyword in filter_values['keywords']:
                    condition = df['product_name'].str.contains(keyword, case=False, na=False)
                    keyword_conditions.append(condition)
                
                if keyword_conditions:
                    return df[pd.concat(keyword_conditions, axis=1).any(axis=1)]
                return df
            
            filtered_df = apply_filter(filtered_df, keyword_filter, "키워드")
            filter_details['keywords'] = {'matched_values': filter_values['keywords']}
        
        # 가격 필터링
        if filter_values.get('price_limit'):
            print(f"\n=== 가격 필터링 ===")
            print(f"가격 제한: {filter_values['price_limit']}")
            print(f"필터링 전 데이터 수: {len(filtered_df)}")
            
            def price_filter(df):
                price_limit = int(filter_values['price_limit'])  # 명시적으로 int로 변환
                df['numeric_price'] = df['current_price'].apply(convert_price_to_number)
                filtered = df[df['numeric_price'] <= price_limit]
                print(f"가격 필터링 결과: {len(filtered)}개")
                print(f"첫 번째 상품 가격: {df['current_price'].iloc[0]} -> {df['numeric_price'].iloc[0]}")
                return filtered
            
            filtered_df = apply_filter(filtered_df, price_filter, "가격")
            filter_details['price_limit'] = {'matched_values': f"{filter_values['price_limit']:,}원 이하"}
        
        # 결과 처리
        total_results = len(filtered_df)
        total_pages = math.ceil(total_results / per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        print(f"필터링 후 결과 수: {total_results}")
        
        paginated_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        results = []
        for idx, row in paginated_df.iterrows():
            try:
                result = {
                    'id': str(idx),
                    'product_name': str(row.get('product_name', '')),
                    'mall_name': str(row.get('mall_name', '')),
                    'current_price': str(row.get('current_price', '')) if not pd.isna(row.get('current_price')) else '',
                    'original_price': str(row.get('original_price', '')) if not pd.isna(row.get('original_price')) else '',
                    'thumbnail_img_url': str(row.get('thumbnail_img_url', '')) if not pd.isna(row.get('thumbnail_img_url')) else '/static/no-image.png',
                    'product_url_path': str(row.get('product_url_path', '')) if not pd.isna(row.get('product_url_path')) else '#'
                }
                results.append(result)
            except Exception as e:
                print(f"결과 처리 중 오류 (인덱스 {idx}): {str(e)}")
                continue
        
        response_data = {
            'results': results,
            'total_count': total_results,
            'current_page': page,
            'total_pages': total_pages,
            'per_page': per_page,
            'filter_details': filter_details
        }
        
        print(f"=== 검색 완료 ===")
        print(f"반환할 결과 수: {len(results)}")
        
        return jsonify(response_data)

    except Exception as e:
        print(f"=== 검색 중 오류 발생 ===")
        print(f"오류 메시지: {str(e)}")
        print(f"상세 오류 정보:\n{traceback.format_exc()}")
        return jsonify({
            'error': '검색 중 오류가 발생했습니다.',
            'message': str(e),
            'results': [],
            'total_count': 0,
            'current_page': 1,
            'total_pages': 0,
            'per_page': per_page,
            'filter_details': {}
        })

def get_ai_recommendations(query, products, top_n=3):
    try:
        if products.empty:
            return {
                "recommendations": [],
                "message": "검색 결과가 없어 추천을 생성할 수 없습니다."
            }

        product_info = []
        for _, product in products.iterrows():
            info = {
                'name': product['product_name'],
                'price': product['current_price'],
                'mall': product['mall_name'],
                'category': product['category'],
                'style': product.get('style', ''),
                'season': product.get('season', ''),
                'color_option': product.get('color_option', ''),
                'thumbnail_img_url': product['thumbnail_img_url'],
                'product_url_path': product['product_url_path']
            }
            product_info.append(info)

        prompt = f"""
사용자 질문: {query}

다음은 검색된 상품 목록입니다:
{json.dumps(product_info, ensure_ascii=False, indent=2)}

위 상품들 중에서 사용자의 요구사항에 가장 적합한 상품 3개를 선택하고, 아래 가이드라인에 따라 상세한 추천 이유와 스타일링 제안을 해주세요:

1. 사용자의 요구사항 분석
   - 검색 의도 파악 (목적, 상황, 선호도 등)
   - 연령대나 성별이 언급된 경우 그에 맞는 스타일 고려
   - 계절이나 날씨가 언급된 경우 적절한 코디 제안

2. 상품 추천 시 포함할 내용
   - 상품의 디자인, 소재, 핏 등 구체적인 특징
   - 가격대비 장점
   - 실제 착용 시 어울리는 체형이나 스타일
   - 다른 아이템과의 코디네이션 제안
   - 구매 시 참고할 사이즈나 컬러 팁

3. 스타일링 제안
   - 추천 상품과 잘 어울리는 다른 아이템 구체적 제안
   - TPO(Time, Place, Occasion)를 고려한 코디 제안
   - 액세서리나 신발 등 포인트 아이템 추천
   - 스타일링 시 주의할 점이나 팁 제공

반드시 아래 형식의 유효한 JSON으로 응답해주세요:

{{
    "recommendations": [
        {{
            "product_name": "상품명",
            "reason": "상세한 추천 이유 (위 가이드라인에 따라 구체적으로 작성)",
            "styling_tip": "스타일링 제안 (코디네이션, 액세서리, 신발 등 구체적 제안)",
            "thumbnail_img_url": "이미지URL",
            "product_url_path": "상품URL",
            "price": "가격",
            "mall_name": "쇼핑몰명"
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant that provides personalized product recommendations. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # 응답 처리 부분 수정
        try:
            content = response.choices[0].message.content.strip()
            # JSON 블록만 추출
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
            
            print("정제된 AI 응답:", content)  # 디버깅을 위한 출력
            
            recommendations = json.loads(content)
            
            # 추천 결과가 비어있는 경우 처리
            if not recommendations.get('recommendations'):
                return {
                    "recommendations": [{
                        "product_name": product_info[0]['name'],
                        "reason": "현재 검색 결과에서 가장 관련성 높은 상품입니다.",
                        "thumbnail_img_url": product_info[0]['thumbnail_img_url'],
                        "product_url_path": product_info[0]['product_url_path'],
                        "price": product_info[0]['price'],
                        "mall_name": product_info[0]['mall']
                    }]
                }
            
            return recommendations
            
        except (json.JSONDecodeError, IndexError) as e:
            print(f"처리 오류: {str(e)}")
            print(f"받은 응답: {content}")
            
            # 최소한 하나의 상품은 반환
            if product_info:
                return {
                    "recommendations": [{
                        "product_name": product_info[0]['name'],
                        "reason": "추천 시스템 오류가 발생했지만, 검색 결과에서 가장 관련성 높은 상품입니다.",
                        "thumbnail_img_url": product_info[0]['thumbnail_img_url'],
                        "product_url_path": product_info[0]['product_url_path'],
                        "price": product_info[0]['price'],
                        "mall_name": product_info[0]['mall']
                    }]
                }
            else:
                return {
                    "recommendations": [],
                    "message": "죄송합니다. 추천할 수 있는 상품이 없습니다."
                }
                
    except Exception as e:
        print(f"AI 추천 생성 중 오류: {str(e)}")
        traceback.print_exc()  # 상세 오류 정보 출력
        return {
            "recommendations": [],
            "message": "추천 생성 중 오류가 발생했습니다."
        }

@app.route('/recommend', methods=['POST'])
def recommend():
    """AI 추천 API"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        products_data = data.get('products', [])
        
        # 데이터프레임으로 변환
        products_df = pd.DataFrame(products_data)
        
        # AI 추천 결과 생성
        recommendations = get_ai_recommendations(query, products_df)
        
        if recommendations:
            return jsonify(recommendations)
        else:
            return jsonify({'error': '추천 생성 중 오류가 발생했습니다.'})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': '추천 처리 중 오류가 발생했습니다.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))