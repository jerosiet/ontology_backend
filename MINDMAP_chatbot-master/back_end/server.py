import os
import fitz # PyMuPDF
import cv2
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import re
from flask import Flask, request, jsonify
from flask_cors import CORS # Keep CORS for cross-origin requests
import tempfile
import shutil
from docx import Document
from PIL import Image
import pytesseract
from io import BytesIO
import nltk # For NLTK setup
from nltk.tokenize import sent_tokenize # For sentence tokenization in OntologyMindmapGenerator (not directly used here but was in original code)

# --- Configuration for Tesseract OCR (if not in PATH) ---
# If tesseract is not in your system's PATH, you might need to specify its path.
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS (if installed via brew):
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# ---------------------------------------------------------

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt', 'docx'}

# --- OpenAI Client Initialization ---
from openai import OpenAI
from dotenv import load_dotenv
import time # For measuring execution time

# Load environment variables from secrect.env
load_dotenv(dotenv_path="secrect.env")

def initialize_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please create a 'secrect.env' file or set the environment variable.")
    return OpenAI(api_key=api_key)

# Initialize the OpenAI client globally for both document processing and chatbot
client = initialize_client()

# --- NLTK Setup ---
import nltk
nltk.download('punkt_tab')
# --- NLTK Setup ---
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError: # This is the correct exception for missing NLTK data
        print("NLTK 'punkt' tokenizer not found, attempting to download...")
        try:
            nltk.download('punkt')
            print("NLTK 'punkt' tokenizer downloaded successfully.")
        except Exception as e:
            print(f"Error downloading NLTK 'punkt' tokenizer: {e}")
            print("Please try running 'python -c \"import nltk; nltk.download(\'punkt\')\"' manually.")
    except Exception as e:
        print(f"An unexpected error occurred during NLTK setup: {e}")

setup_nltk() # Call NLTK setup on app startup

# --- Ontology and Embedding Model Setup (from LLMquery.py) ---
from owlready2 import *

ontology_path = "static/MINDMAP.owl"
name_ontology = ontology_path.split('/')[-1].split('.')[0]

# Check if ontology file exists
if not os.path.exists(ontology_path):
    # Create a dummy ontology if not found, for initial setup guidance
    print(f"WARNING: '{ontology_path}' not found. Creating a dummy OWL file for demonstration.")
    try:
        with get_ontology("http://www.semanticweb.org/MINDMAP#") as onto_dummy:
            class TestClass(Thing):
                comment = ["This is a test class for the dummy ontology."]
            class TestInstance(TestClass):
                summary = ["This is a dummy summary for testing purposes."]
            onto_dummy.save(file=ontology_path, format="rdfxml")
        print("Dummy MINDMAP.owl created. Replace it with your actual ontology for full functionality.")
    except Exception as e:
        print(f"Error creating dummy ontology: {e}")
        # If dummy creation fails, the app might not start or ontology-related functions will fail
        pass # Allow app to try loading it, which will likely fail later

try:
    onto = get_ontology(ontology_path).load()
except Exception as e:
    raise Exception(f"Failed to load ontology from {ontology_path}: {e}. Ensure the file is valid and accessible.")

# Note: "model_embedding" is a placeholder. You need to replace this with a valid
# SentenceTransformer model name (e.g., "all-MiniLM-L6-v2") or a path to a local model.
# Ensure the model is downloaded or accessible.
try:
    from sentence_transformers import SentenceTransformer
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2") # Example model
except ImportError:
    print("Warning: 'sentence-transformers' not installed. Using OpenAI embeddings for similarity search.")
    model_embedding = None
except Exception as e:
    print(f"Warning: Failed to load SentenceTransformer model. Embedding functionality may be limited: {e}")
    model_embedding = None


# --- Global variables for chatbot ---
chat_histories = {}
user_id = '234' # In a real application, this should be dynamically generated or managed per user session


# --- PDF processing functions (adapted from your code) ---
class PDFProcessor:
    def __init__(self, client):
        self.client = client

    def extract_text_from_pdf(self, pdf_path, start_page, end_page):
        doc = fitz.open(pdf_path)
        list_paragraphs = []
        paragraph = ''

        # Ensure page range is valid
        start_page = max(1, start_page)
        end_page = min(doc.page_count, end_page)

        for page_num in range(start_page - 1, end_page):
            page = doc[page_num]

            # Extract page image for layout analysis
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            edged = cv2.Canny(blurred, 10, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            horizontal_lines = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter for horizontal lines (aspect ratio > 40)
                if w / (h + 1e-6) > 40: # Add epsilon to avoid division by zero
                    horizontal_lines.append([x, y, w, h])

            horizontal_lines = sorted(horizontal_lines, key=lambda item: item[1])

            width, height = page.rect.width, page.rect.height
            # Define coordinates for text extraction based on detected lines
            if len(horizontal_lines) > 1:
                y0 = horizontal_lines[0][1]
                y1 = horizontal_lines[-1][1]
                # Adjust coordinates to fit page dimensions and avoid excessive clipping
                coords = (0, y0 * (height / img.shape[0]) + 10, width, y1 * (height / img.shape[0]) - 5)
                paragraph = self._extract_text_page(page, paragraph, list_paragraphs, coords)
            elif len(horizontal_lines) == 1:
                x, y, w, h = horizontal_lines[0]
                scaled_y = y * (height / img.shape[0])
                if scaled_y > height / 2:
                    coords = (0, 0, width, scaled_y - 5)
                else:
                    coords = (0, scaled_y + 10, width, height - 50)
                paragraph = self._extract_text_page(page, paragraph, list_paragraphs, coords)
            else:
                coords = (0, 0, width, height - 50)
                paragraph = self._extract_text_page(page, paragraph, list_paragraphs, coords)

        doc.close()
        return list_paragraphs

    def _extract_text_page(self, page, paragraph, list_para, coordinates):
        # Find coordinates of title text based on font properties
        title_y_coordinates = []
        try:
            block_dict = page.get_text('dict', clip=coordinates)
            for spans in block_dict.get('blocks', []):
                for line in spans.get('lines', []):
                    if line.get('spans'):
                        font = line['spans'][0]['font']
                        # Check for common bold/italic font indicators
                        if 'Bold' in font or 'Italic' in font:
                            title_y_coordinates.append(line['bbox'][1])
        except Exception as e:
            print(f"Error getting text dict for title detection: {e}")
            # Fallback if dict extraction fails or is empty
            title_y_coordinates = []


        # Extract text blocks
        try:
            blocks = page.get_text('blocks', clip=coordinates)
            for b in blocks:
                # b[1] is y0 coordinate of the block
                if b[1] not in title_y_coordinates:  # Regular text
                    text = b[4].replace('.\n', '.#')
                    text = text.replace('\n', ' ')
                    if text.endswith('.#'):
                        paragraph += text
                        list_para.append(paragraph.strip())
                        paragraph = ''
                    else:
                        paragraph += text
                else:  # Title text
                    text = b[4].replace('.\n', '.#')
                    text = text.replace('\n', ' ')
                    list_para.append('<TITLE>_' + text.upper().strip())
        except Exception as e:
            print(f"Error getting text blocks: {e}")

        return paragraph

    def preprocess_paragraphs(self, list_paragraphs):
        new_list_paragraphs = []
        flag = 0
        idx_count = 0

        for idx, para in enumerate(list_paragraphs):
            if para.startswith('<TITLE>_'):
                if flag == 0 and idx_count > 0: # If previous was content, add it before new title sequence
                    new_list_paragraphs.append(''.join(list_paragraphs[idx - idx_count : idx]))
                    idx_count = 0
                flag = 1
                idx_count += 1
            else:
                if flag == 1: # End of a title sequence, combine and add
                    combined_title = ''.join(list_paragraphs[idx - idx_count : idx])
                    new_list_paragraphs.append(combined_title)
                    idx_count = 0
                flag = 0
                new_list_paragraphs.append(para)

        # Handle any remaining combined titles or content at the end
        if idx_count > 0:
            new_list_paragraphs.append(''.join(list_paragraphs[len(list_paragraphs) - idx_count : len(list_paragraphs)]))

        return [p for p in new_list_paragraphs if p.strip()] # Filter out empty strings


# --- Text Analysis functions (adapted from your code) ---
class TextAnalyzer:
    def __init__(self, client):
        self.client = client
        self.original_prompt_template = '''Bạn là chuyên gia trong việc phân tích level cho mục lục.
                        Từ đoạn văn bản nghi nghờ của người dùng đưa vào nhận biết nó có phải là tiêu đề hay không.
                        Nếu là tiêu đề thì phân cấp và trả về kết quả dưới dạng JSON với cấu trúc sau:

                        {{
                          "level 1": "(tiêu đề cấp cao nhất: 'Chương' hoặc tiêu đề tương đương, cùng với nội dung tiêu đề theo sau)",
                          "level 2": "(tiêu đề cấp tiếp theo: Mục lớn như I , II , L , IL,... . hoặc dạng số la mã, cùng với nội dung tiêu đề theo sau)",
                          "level 3": "(tiêu đề cấp tiếp theo: Mục, như 1., 2. ,... hoặc các định dạng chữ số nguyên, cùng với nội dung tiêu đề theo sau)",
                          "level 4": "(tiêu đề cấp tiếp theo: Ví dụ: A. , B., C. ,... cùng với nội dung tiêu đề theo sau)",
                          "level 5": "(tiêu đề cấp tiếp theo: ví dụ: a) , b) , c) ,... cùng với nội dung tiêu đề theo sau)",
                          ...
                        }}

                        Căn cứ mục lục này để phân cấp level tiếp theo:

                        {level_document}

                        Yêu cầu:
                        - Bỏ qua thẻ `<TITLE>_` ở đầu câu.
                        LƯU Ý: KHÔNG ĐƯỢC BỎ THÔNG TIN NÀO TRONG ĐOẠN VĂN ĐƯỢC CUNG CẤP
                        Đầu ra:
                        - DỰA VÀO NGỮ CẢNH ĐỂ SỬA LỖI CHÍNH TẢ NẾU CÓ.
                        - Nếu đoạn là tiêu đề hợp lệ, trả về JSON với phân cấp rõ ràng.
                        - Nếu đoạn không phải tiêu đề, trả về `None`.
                        - KHÔNG GHI THÊM BẤT CỨ THÔNG TIN NÀO KHÁC NGOÀI JSON HOẶC None.
                        '''

    def increase_level(self, text):
        match = re.search(r'(\d+)', text)
        num = 0
        if match:
            num = int(match.group(1)) + 1
            return text.replace(match.group(1), str(num)), num
        return text, num

    def create_dataframe(self, list_processed_paragraphs):
        prompt_template = self.original_prompt_template.format(level_document='')
        level_document = ''

        data_records = [] # Danh sách chứa các dictionary, mỗi dictionary là một hàng trong DataFrame
        # Lưu trữ đường dẫn các tiêu đề hiện tại đang active
        current_heading_context = {f'level {i}': None for i in range(1, 6)} 

        for para in list_processed_paragraphs:
            if para.startswith('<TITLE>_'):
                clean_para = para[8:].strip()
                try:
                    response = self.client.chat.completions.create(
                        model='gpt-4o-mini',
                        temperature=0,
                        messages=[
                            {"role": "system", "content": prompt_template},
                            {"role": "user", "content": clean_para}
                        ]
                    )
                    title_json_str = response.choices[0].message.content.strip()

                    if title_json_str.lower() != 'none':
                        try:
                            new_heading_info = json.loads(title_json_str)
                            
                            # Xác định cấp độ cao nhất mà LLM đã cập nhật
                            highest_llm_level = 0
                            for k in new_heading_info.keys():
                                match = re.search(r'level (\d+)', k)
                                if match:
                                    highest_llm_level = max(highest_llm_level, int(match.group(1)))
                            
                            # Cập nhật context tiêu đề hiện tại
                            for i in range(1, 6):
                                level_key = f'level {i}'
                                if level_key in new_heading_info:
                                    current_heading_context[level_key] = new_heading_info[level_key]
                                elif i > highest_llm_level and highest_llm_level > 0:
                                    # Nếu một tiêu đề cấp cao hơn được cập nhật, các cấp thấp hơn sẽ bị xóa
                                    current_heading_context[level_key] = None
                                # else: Giữ nguyên các cấp cao hơn nếu không được LLM cập nhật

                            # Thêm một hàng vào data_records để đánh dấu đây là một tiêu đề mới
                            # Hàng này sẽ chỉ chứa thông tin tiêu đề, không có nội dung văn bản.
                            row_for_heading = current_heading_context.copy()
                            # Đánh dấu hàng này là loại 'heading' để build_tree biết
                            row_for_heading['type'] = 'heading' 
                            data_records.append(row_for_heading)

                            # Cập nhật prompt cho các lần gọi LLM tiếp theo
                            level_document += title_json_str + "\n"
                            prompt_template = self.original_prompt_template.format(level_document=level_document)

                        except json.JSONDecodeError:
                            print(f"Warning: LLM returned invalid JSON for title: {title_json_str}")
                            # Xử lý như nội dung bình thường nếu LLM trả về JSON không hợp lệ
                            content_row = current_heading_context.copy()
                            content_row['content'] = clean_para
                            content_row['type'] = 'content'
                            data_records.append(content_row)
                    else: # LLM trả về 'None' (không phải tiêu đề)
                        # Xử lý như nội dung bình thường
                        content_row = current_heading_context.copy()
                        content_row['content'] = clean_para
                        content_row['type'] = 'content'
                        data_records.append(content_row)
                except Exception as e:
                    print(f"Error calling LLM for title detection: {e}")
                    # Xử lý dự phòng: coi đây là nội dung bình thường
                    content_row = current_heading_context.copy()
                    content_row['content'] = clean_para
                    content_row['type'] = 'content'
                    data_records.append(content_row)
            else: # Đoạn văn bản nội dung thông thường (không bắt đầu bằng <TITLE>_)
                # Xử lý như nội dung bình thường, gán vào context tiêu đề hiện tại
                content_row = current_heading_context.copy()
                content_row['content'] = para # Giữ nguyên para, có thể có khoảng trắng đầu/cuối
                content_row['type'] = 'content'
                data_records.append(content_row)
        
        # Chuyển đổi danh sách các dictionary thành DataFrame
        df = pd.DataFrame(data_records)
        # Bỏ các hàng hoàn toàn rỗng (nếu có)
        df.dropna(how='all', subset=[f'level {i}' for i in range(1,6)] + ['content', 'type'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def build_tree(self, df):
        tree = {}

        # Hàm trợ giúp để truy cập/tạo các dictionary lồng nhau trong cây
        def get_nested_node(current_node, path_elements):
            node = current_node
            for element in path_elements:
                if element not in node:
                    node[element] = {} # Khởi tạo một tiêu đề mới là một dictionary
                node = node[element]
            return node

        # Duy trì đường dẫn tiêu đề hiện tại (dạng list các string tiêu đề)
        current_active_headings_path = [] 

        for _, row in df.iterrows():
            row_type = row.get('type') # Lấy loại hàng: 'heading' hoặc 'content'

            # Luôn cập nhật current_active_headings_path dựa trên các cột level_X
            # Điều này đảm bảo khi một hàng nội dung xuất hiện, nó biết mình thuộc về tiêu đề nào
            temp_path = []
            for i in range(1, 6):
                level_key = f'level {i}'
                heading = row.get(level_key)
                if heading is not None:
                    # Loại bỏ '<TITLE>_' và khoảng trắng nếu có
                    clean_heading = str(heading).replace('<TITLE>_', '').strip()
                    temp_path.append(clean_heading)
                else:
                    break
            current_active_headings_path = temp_path # Cập nhật đường dẫn cho hàng hiện tại

            if row_type == 'heading':
                # Hàng này chỉ định nghĩa một tiêu đề mới.
                # Đảm bảo node cho tiêu đề này tồn tại trong cây.
                if current_active_headings_path:
                    get_nested_node(tree, current_active_headings_path) 
            elif row_type == 'content' and 'content' in row and row['content']:
                # Hàng này chứa nội dung. Thêm nội dung vào node tiêu đề active hiện tại.
                if current_active_headings_path:
                    target_node = get_nested_node(tree, current_active_headings_path)
                    
                    if "_content" not in target_node:
                        target_node["_content"] = []
                    target_node["_content"].append(row['content'].strip())
                else:
                    print(f"Warning: Content found without an active heading path: {row['content']}")
                    # Bạn có thể xử lý nội dung không có tiêu đề ở đây (ví dụ: gán vào một root_content list)

        # Bước hậu xử lý cuối cùng để làm sạch cây (giống như trước)
        def post_process_tree(node):
            if isinstance(node, dict):
                cleaned_node = {}
                has_sub_children = False # Theo dõi xem node này có các tiêu đề con không

                for k, v in node.items():
                    if k == "_content":
                        if v: # Chỉ giữ _content nếu nó không rỗng
                            cleaned_node[k] = v
                    else: # Đây là một tiêu đề con
                        processed_sub = post_process_tree(v)
                        if processed_sub: # Chỉ bao gồm nếu tiêu đề con có nội dung/tiêu đề con
                            cleaned_node[k] = processed_sub
                            has_sub_children = True
                
                # Nếu một node chỉ có _content và không có tiêu đề con, đơn giản hóa nó thành danh sách _content
                if "_content" in cleaned_node and not has_sub_children:
                    return cleaned_node["_content"]
                
                return cleaned_node
            # Nếu node là list (ví dụ: đã được đơn giản hóa từ _content), trả về nguyên trạng
            return node

        return post_process_tree(tree)


# --- Chatbot Functions (from LLMquery.py) ---
# These functions are already imported and adapted from your previous LLMquery.py
# They will use the globally initialized 'client' and 'onto' objects.

def get_entities_with_annotation(onto, annotation):
    """
    Tìm tất cả các instances và classes có annotation trong ontology.

    Parameters:
    - onto: Ontology đang làm việc
    - annotation: Tên annotation cần tìm

    Returns:
    - Dictionary chứa tất cả các instances và classes có annotation
    """
    result = {}
    processed_entities = set()  # Tập hợp để theo dõi các entity đã xử lý

    # Duyệt qua tất cả các lớp trong ontology
    for cls in onto.classes():
        # Kiểm tra và thêm annotation của class (nếu có và chưa xử lý)
        if cls.name not in processed_entities and hasattr(cls, annotation):
            annotation_value = getattr(cls, annotation)
            if annotation_value:  # Kiểm tra không rỗng
                # Loại bỏ các giá trị trùng lặp trong annotation_value nếu nó là list
                if isinstance(annotation_value, list):
                    # Sử dụng dict.fromkeys để loại bỏ các phần tử trùng lặp nhưng giữ thứ tự
                    unique_values = list(dict.fromkeys(tuple(x) if isinstance(x, list) else x for x in annotation_value))
                    result[cls.name] = {annotation: unique_values}
                else:
                    result[cls.name] = {annotation: annotation_value}
                processed_entities.add(cls.name)

        # Xử lý các instances
        for instance in cls.instances():
            if instance.name not in processed_entities and hasattr(instance, annotation):
                annotation_value = getattr(instance, annotation)
                if annotation_value:
                    # Loại bỏ các giá trị trùng lặp trong annotation_value nếu nó là list
                    if isinstance(annotation_value, list):
                        # Chuyển đổi các phần tử để có thể so sánh và loại bỏ trùng lặp
                        unique_values = list(dict.fromkeys(tuple(x) if isinstance(x, list) else x for x in annotation_value))
                        result[instance.name] = {annotation: unique_values}
                    else:
                        result[instance.name] = {annotation: annotation_value}
                    processed_entities.add(instance.name)
    return result


def find_relation(onto):
    """
    Tìm cấu trúc cây phân cấp của ontology và các instances.

    Parameters:
    - onto: Ontology đang làm việc

    Returns:
    - Dictionary chứa cấu trúc cây ontology
    """
    result = {}

    # Danh sách tất cả các lớp để kiểm tra
    all_classes = list(onto.classes())
    # Tìm các top-level classes (không có superclass ngoài Thing)
    top_classes = []
    for cls in all_classes:
        # Lấy tất cả các lớp cha trực tiếp
        parents = [p for p in cls.is_a if isinstance(p, onto.Thing.__class__)]
        # Loại bỏ Thing hoặc lớp chính nó
        parents = [p for p in parents if p != onto.Thing and p != cls]

        if not parents:  # Nếu không có lớp cha nào ngoài Thing
            top_classes.append(cls)
    def build_tree(cls):
        try:
          label = cls.label[0]
          if label:
            node_name = label
          else:
            node_name = cls.name
        except:
          node_name = cls.name
        """Xây dựng cây cho một lớp cụ thể"""
        node = {"name": node_name}

        # Tìm tất cả lớp con trực tiếp
        direct_subclasses = []
        for sub in all_classes:
            if cls in sub.is_a and sub != cls:
                direct_subclasses.append(sub)

        if direct_subclasses:  # Nếu có lớp con
            node["subclasses"] = []
            for sub in direct_subclasses:
                node["subclasses"].append(build_tree(sub))
        else:  # Nếu là lớp lá (không có lớp con)
            # Tìm instances của lớp này
            instances = list(cls.instances())
            if instances:
                node["Instances"] = [inst.name for inst in instances]
        return node

    for cls in top_classes:
        result[cls.name] = build_tree(cls)

    return result


def create_explication(entities_with_annotation_sumarry : dict):
  explication = {}
  for entity, entity_value in entities_with_annotation_sumarry.items():
    for annotation, value in entity_value.items():
        full_text_information = ''.join(value)
        explication[entity] = full_text_information
  return explication


def find_entities_from_question_PP1(relation, explication, question, chat_history):
    messages = [
        {
            "role": "system",
            "content": """Bạn là một agent hữu ích được thiết kế để tìm tên thực thể liên quan đến câu hỏi.

            Dựa trên TÊN THỰC THỂ, CÁC QUAN HỆ TƯƠNG ỨNG và THÔNG TIN CHÚ THÍCH, hãy xác định xem câu hỏi của người dùng có thể được trả lời bằng các thực thể đã liệt kê hay không.

            - Trích xuất các thực thể được nhắc đến trực tiếp trong  câu hỏi hoặc
            - Tìm ra những thực thể phù hợp có khả năng trả lời cho câu hỏi, ngay cả khi người dùng không nhắc đến tên thực thể rõ ràng.
            - Ưu tiên các thực thể có mối liên hệ ngữ nghĩa hoặc quan hệ tri thức gần với nội dung câu hỏi.
            - Nếu thông tin chú thích trùng thì trả về tên thực thể cấp thấp nhất.
            Kết quả:
            Trả về kết quả là các class dưới dạng JSON như sau:
            {
                "(parent_class)": [danh sách thực thể]
            }


            Ví dụ:
            Câu hỏi: "Hiệp ước Pháp - Hoa ký ngày 28-2-1946 có nội dung gì và tình hình sau đó đã đặt Việt Nam Dân chủ Cộng hòa trước lựa chọn nào?"
            Đáp án:
            {
                "Hiệp_định_sơ_bộ_1946_1": ["Hiệp_ước_Pháp_-_Hoa"]
            }

            Câu hỏi: "Sau Cách mạng tháng Tám, chính phủ Việt Nam Dân chủ Cộng hòa đã thực hiện những biện pháp gì để khôi phục kinh tế và giải quyết nạn đói?"
            Đáp án:
            {
                "Khôi_phục_kinh_tế_11": ["Khôi_phục_kinh_tế_111"]
            }
            Câu hỏi: "Chính phủ Việt Nam Dân chủ Cộng hòa đã có những biện pháp gì để đối phó với khó khăn về chính trị, kinh tế và ngoại giao trong năm 1945–1946"
            Đáp án:
            {
                "Khôi_phục_kinh_tế_11": ["Khôi_phục_kinh_tế_111"],
                "Hiệp_định_sơ_bộ_1946_1": ["Tiếp_xúc_Việt-Pháp_1945", "Hiệp_ước_Pháp_-_Hoa"]
            }
            Nếu câu hỏi không có thông tin liên quan, trả về:
            {
                "Trong": []
            }
            """
        },
        {
            "role": "user",
            "content": f"CÁC THỰC THỂ VÀ CÁC QUAN HỆ TƯƠNG ỨNG:\n{json.dumps(relation, indent=4, ensure_ascii=False)}"
        },
        {
            "role": "user",
            "content": f"TÊN THỰC THỂ VÀ THÔNG TIN CHÚ THÍCH:\n{json.dumps(explication, indent=4, ensure_ascii=False)}"
        },
        {
            "role": "user",
            "content": f"THÔNG TIN NGỮ CẢNH:\n{chat_history}"
        },
        {
            "role": "user",
            "content": f"CÂU HỎI:\n{question}"
        }
    ]

    response = client.chat.completions.create(
        model='gpt-4o',
        temperature=0,
        messages=messages
    )

    return response.choices[0].message.content

def get_direct_class_of_individual(onto, individual_name):
    """
    Trả về class cha trực tiếp đầu tiên (rdf:type) của một individual.

    Args:
        onto: Ontology đã load bằng Owlready2.
        individual_name (str): Tên của individual trong ontology.

    Returns:
        str hoặc None: Tên class cha trực tiếp, hoặc None nếu không có.
    """
    try:
        individual = onto.search_one(iri="*" + individual_name) # Use search_one as individual_name might not be direct key
        if individual and individual.is_a:
            # Filter out owl:Thing and other non-specific parent classes if needed
            specific_parents = [p for p in individual.is_a if isinstance(p, ThingClass) and p != owl.Thing]
            if specific_parents:
                return specific_parents[0].name
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"[!] Lỗi khi truy xuất class cha của individual '{individual_name}': {e}")
        return None


def query_all(name_ontology, query_all_class_info, value):
    prefix = '''
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX mindmap: <http://www.semanticweb.org/MINDMAP#>
    '''

    class_name = str(value).replace(f"{name_ontology}.", "")

    # Truy vấn tất cả thông tin của class (predicate và giá trị)
    query1 = f'''
        SELECT ?summary
        WHERE {{
          mindmap:{class_name} mindmap:summary ?summary .
        }}
    '''
    query_all_class_info.append(prefix + query1)


def create_query(name_ontology, json_data):
    """
    Tổng quát hóa hàm tạo query từ ontology dựa trên json_data.

    Args:
        name_ontology: Tên ontology đang làm việc
        json_data: Dữ liệu JSON chứa các key và list các entity cần truy vấn

    Returns:
        list: Tất cả thông tin truy vấn được
    """
    query_all_class_info = []

    for key, values in json_data.items():
        has_class_parent = False
        set_class_parent = set()
        if key == "Trong":
            continue  # Bỏ qua nếu key là 'Trong'

        for value in values:
            # Lấy thực thể từ ontology
            entity = onto.search_one(iri="*" + value)
            print(f"Searching for entity: {value}, Found: {entity}")
            if entity is None:
                print(f"[!] Không tìm thấy '{value}' trong ontology.")
                continue

            # Nếu là class, xử lý bình thường
            if isinstance(entity, ThingClass):
                children = onto.get_children_of(entity)
                print(f"Children of {entity.name}: {[c.name for c in children]}")
                query_all(name_ontology, query_all_class_info, entity)
                for child in children:
                    query_all(name_ontology, query_all_class_info, child)
            # Nếu là individual
            else:
                class_parent = get_direct_class_of_individual(onto, value)
                if class_parent:
                    set_class_parent.add(class_parent)
                    has_class_parent = True
        if has_class_parent:
            for class_parent in set_class_parent:
                query_all(name_ontology, query_all_class_info, onto.search_one(iri="*" + class_parent)) # Pass the class object

    return query_all_class_info


def find_question_info(name_ontology, list_query):
    """
    Chạy danh sách SPARQL query và trích xuất kết quả ra dạng text dễ đọc.

    Args:
        name_ontology: Tên ontology (dùng để bỏ prefix khi cần)
        list_query: List query SPARQL cần thực thi

    Returns:
        list: Các kết quả đã tiền xử lý (list of list)
    """
    results = []

    for query in list_query:
        try:
            query_result = list(default_world.sparql_query(query))
        except Exception as e:
            print(f"[!] Lỗi khi thực thi query: {e}")
            continue

        for record in query_result:
            information = []
            for val in record:
                processed_text = ""

                # Nếu có label thì ưu tiên lấy label
                if hasattr(val, "label") and val.label:
                    processed_text = str(val.label[0])
                else:
                    # Nếu không có label, thì xử lý text bình thường
                    processed_text = str(val)
                    if processed_text.startswith(f"{name_ontology}."):
                        processed_text = processed_text.replace(f"{name_ontology}.", "")

                    # Xử lý đặc biệt nếu cần (ví dụ trường hợp '9' bạn từng gặp)
                    if processed_text == '9':
                        processed_text = 'Là con của'

                information.append(processed_text)

            results.append(information)

    return results

def generate_response(relationship ,question_info, question, history ):
  system_prompt  = f'''
            Bạn là một agent hữu ích giúp trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.
            Dựa vào lịch sử cuộc trò chuyện để hiểu rõ hơn ngữ cảnh cuộc trò chuyện.
            THÔNG TIN:

            {question_info}

            LỊCH SỬ CUỘC TRÒ CHUYỆN:

            {history}

            Nếu không có câu trả lời, hãy nói: Tôi không biết, tôi chưa có kiến thức để trả lời câu hỏi này.
  '''
  response = client.chat.completions.create(
      model='gpt-4o-mini',
      temperature=0,
      messages=[
          {
              "role": "system",
              "content": system_prompt
          },
          {
              "role": "user",
              "content": question
          }
          ]
      )
  return response.choices[0].message.content

def get_embedding(text, model = "text-embedding-ada-002"):
    if model_embedding: # Use SentenceTransformer if loaded
        if isinstance(text, str):
            text = [text] # Ensure it's a list for SentenceTransformer
        return [model_embedding.encode(t).tolist() for t in text]
    else: # Fallback to OpenAI if SentenceTransformer not available
        response = client.embeddings.create(
            input = text, # Input should be a list of strings
            model = model
        )
        return [d.embedding for d in response.data]

def find_similar_info_from_raw_informations(question, result_from_ontology, k = 5):
    if not result_from_ontology:
        return ["Không có thông tin liên quan từ ontology."]

    # Ensure result_from_ontology contains strings for embedding
    flat_results = []
    for sublist in result_from_ontology:
        if isinstance(sublist, list):
            flat_results.extend([str(item) for item in sublist if item is not None])
        elif sublist is not None:
            flat_results.append(str(sublist))

    if not flat_results:
        return ["Không có thông tin liên quan từ ontology sau khi làm phẳng."]

    embeddings_list = get_embedding(flat_results)
    embeddings = np.array(embeddings_list, dtype=np.float32)

    dimension = embeddings.shape[1]
    import faiss # Ensure faiss is imported here if not global
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    similar_info = []

    # Get embedding for the question (which is a single string)
    question_embedding = get_embedding([question])[0] # Get the single embedding from the list
    query_embedding = np.array(question_embedding, dtype=np.float32).reshape(1, -1) # Reshape for FAISS search

    distances, indices = index.search(query_embedding, min(k, len(flat_results))) # Ensure k doesn't exceed available items

    # Retrieve the original raw information strings based on indices
    for idx in indices[0]:
        similar_info.append(flat_results[idx])

    return similar_info


# --- Helper function for file type validation ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoint for Document Upload and Processing ---
@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    temp_dir = None
    try:
        # Create a temporary directory to save the uploaded file
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath) # Save the uploaded file

        file_extension = file.filename.rsplit('.', 1)[1].lower()
        extracted_paragraphs = []

        pdf_processor = PDFProcessor(client) # Initialize with the global client

        # Extract text based on file type
        if file_extension == 'pdf':
            doc = fitz.open(filepath)
            total_pages = doc.page_count
            doc.close() # Close to release file handle before re-opening in extract_text_from_pdf
            extracted_paragraphs = pdf_processor.extract_text_from_pdf(filepath, 1, total_pages)
        elif file_extension in ['png', 'jpg', 'jpeg']:
            try:
                img = Image.open(filepath)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                text = pytesseract.image_to_string(img, lang='vie') # Assuming Vietnamese text
                extracted_paragraphs = [text] if text.strip() else []
            except Exception as e:
                print(f"Pytesseract OCR failed for image: {e}. Attempting with PyMuPDF as fallback.")
                # Fallback to PyMuPDF for image (it can extract text from images too)
                doc = fitz.open(filepath)
                page = doc[0]
                text = page.get_text()
                doc.close()
                extracted_paragraphs = [text] if text.strip() else []
        elif file_extension == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                extracted_paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        elif file_extension == 'docx':
            doc = Document(filepath)
            extracted_paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        if not extracted_paragraphs:
            return jsonify({"error": "No text could be extracted from the document."}), 422 # Unprocessable Entity

        # Preprocess paragraphs and analyze text
        list_processed_paragraphs = pdf_processor.preprocess_paragraphs(extracted_paragraphs)
        text_analyzer = TextAnalyzer(client) # Initialize with the global client
        dataframe = text_analyzer.create_dataframe(list_processed_paragraphs)
        json_tree = text_analyzer.build_tree(dataframe)

        # Return the JSON structure
        return jsonify(json_tree), 200

    except Exception as e:
        print(f"Error during file processing: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    finally:
        # Clean up the temporary directory and file
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Chatbot Endpoint ---

# --- Ontology generator ---
import nltk
from nltk.tokenize import sent_tokenize
import networkx as nx
import types
from collections import defaultdict

# Add this class to your existing code (after your existing classes)
class OntologyMindmapGenerator:
    def __init__(self, client):
        """Initialize the ontology mindmap generator with existing OpenAI client"""
        self.client = client
        
        # Setup sentence tokenizer
        self.setup_nltk()
        
        # Initialize ontology
        onto_path.append('.')
        self.onto = None

    def setup_nltk(self):
        """Setup NLTK tokenizer"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt')
                nltk.download('punkt_tab')
            except Exception as e:
                print(f"Error downloading NLTK data: {e}")

    def is_multi_sentence(self, text):
        """Check if text contains multiple sentences"""
        if not text or not isinstance(text, str):
            return False
        sentences = sent_tokenize(text)
        return len(sentences) > 1

    def extract_keyword(self, text):
        """Extract the most relevant keyword from text using OpenAI"""
        if not self.client:
            words = text.split()
            return ' '.join(words[:2]).title()

        prompt = ("Extract a single keyword or short phrase (1-3 words) that best represents the main "
                  "topic of this text. The keyword must be no longer than 4 words. "
                  "The keyword should be suitable as a class name in an ontology. "
                  "Return ONLY the keyword, nothing else.")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            keyword = response.choices[0].message.content.strip()
            keyword = re.sub(r'[^\w\s]', '', keyword)
            return keyword.title()
        except Exception as e:
            print(f"Error extracting keyword: {e}")
            words = text.split()
            return ' '.join(words[:2]).title()

    def generate_summary(self, text):
        """Generate a Vietnamese concise summary suitable for an annotation"""
        if not self.client:
            return text[:100] + '...' if len(text) > 100 else text

        prompt = ("Generate a Vietnamese concise summary (2-3 sentences) of the following text. "
                  "The summary will be used as an annotation in an ontology. Focus on the key concepts and relationships.")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return text[:100] + '...' if len(text) > 100 else text

    def find_topic_keyword(self, texts):
        """Find an overarching topic keyword for a set of texts"""
        combined = " ".join(texts)
        if not self.client:
            words = combined.split()
            return words[0].title() if words else 'Topic'

        prompt = ("Given the following collection of related text segments, identify ONE overarching "
                  "topic or concept that could serve as a parent class. Return ONLY the topic name, nothing else.")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": combined}
                ]
            )
            topic = response.choices[0].message.content.strip()
            topic = re.sub(r'[^\w\s]', '', topic)
            return topic.title()
        except Exception as e:
            print(f"Error finding topic keyword: {e}")
            words = combined.split()
            return words[0].title() if words else 'Topic'

    def extract_segments_from_json(self, data):
        """Extract multi-sentence segments from hierarchical JSON data"""
        segments = []
        
        def recurse(node, path=None):
            if path is None:
                path = []
            
            if isinstance(node, dict):
                for key, val in node.items():
                    if self.is_multi_sentence(key):
                        segments.append({'path': path, 'text': key})
                    recurse(val, path + [key])
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, str) and self.is_multi_sentence(item):
                        segments.append({'path': path, 'text': item})
                    else:
                        recurse(item, path)
            elif isinstance(node, str) and self.is_multi_sentence(node):
                segments.append({'path': path, 'text': node})
        
        recurse(data)
        return segments

    def build_ontology_with_owlready(self, segments):
        """Build ontology from segments using owlready2"""
        # Create new ontology
        self.onto = get_ontology("http://example.org/DocumentOntology.owl")
        
        with self.onto:
            # Create base class
            DocumentConcept = types.new_class('DocumentConcept', (Thing,))

            # Data properties
            hasText = types.new_class('hasText', (DataProperty,))
            hasText.domain = [DocumentConcept]
            hasText.range = [str]

            summary = types.new_class('summary', (DataProperty,))
            summary.domain = [DocumentConcept]
            summary.range = [str]

            hasPath = types.new_class('hasPath', (DataProperty,))
            hasPath.domain = [DocumentConcept]
            hasPath.range = [str]

            # Object properties
            contains = types.new_class('contains', (ObjectProperty,))
            contains.domain = [DocumentConcept]
            contains.range = [DocumentConcept]

            isRelatedTo = types.new_class('isRelatedTo', (ObjectProperty,))
            isRelatedTo.domain = [DocumentConcept]
            isRelatedTo.range = [DocumentConcept]

            # Group by path
            groups = defaultdict(list)
            for seg in segments:
                groups[tuple(seg['path'])].append(seg['text'])

            created = {}
            for path_key, texts in groups.items():
                if not texts:
                    continue
                
                topic = self.find_topic_keyword(texts).replace(' ', '_')
                if topic not in created:
                    TopicCls = types.new_class(topic, (DocumentConcept,))
                    TopicCls.comment = [f'Topic grouping for {path_key}']
                    created[topic] = TopicCls
                else:
                    TopicCls = created[topic]

                for text in texts:
                    name = self.extract_keyword(text).replace(' ', '_')
                    text_summary = self.generate_summary(text)
                    
                    if name not in created:
                        SegCls = types.new_class(name, (TopicCls,))
                        SegCls.comment = [text_summary]
                        SegCls.hasText = [text]
                        SegCls.summary = [text_summary]
                        SegCls.hasPath = [str(path_key)]
                        created[name] = SegCls

        return self.onto

    def process_json_data(self, json_data, output_filename="generated_ontology"):
        """Process JSON data, build ontology, and save to file"""
        segments = self.extract_segments_from_json(json_data)
        print(f'Found {len(segments)} segments')
        
        if not segments:
            return None, "No multi-sentence segments found in the document"
        
        self.onto = self.build_ontology_with_owlready(segments)
        print(f'Created {len(list(self.onto.classes()))} classes')
        
        # Save ontology to static directory
        output_path = os.path.join('static', f'{output_filename}.owl')
        self.onto.save(file=output_path, format='rdfxml')
        print(f'Saved ontology to {output_path}')
        
        return output_path, f"Ontology created with {len(list(self.onto.classes()))} classes"

def extract_ontology_info(ontology_path):
    """Extract information from OWL file and return as simple JSON with index, parent_index, name, summary"""
    try:
        # Load the ontology
        onto = get_ontology(ontology_path).load()
        
        # Create a simple list structure
        nodes = []
        class_to_index = {}
        current_index = 0
        
        # First pass: collect all classes and assign indices
        all_classes = list(onto.classes())
        
        for cls in all_classes:
            if cls.name == 'Thing':  # Skip the basic Thing class
                continue
                
            class_to_index[cls.name] = current_index
            current_index += 1
        
        # Second pass: build the node structure
        for cls in all_classes:
            if cls.name == 'Thing':  # Skip the basic Thing class
                continue
            
            # Get parent classes (excluding Thing)
            parent_classes = [p for p in cls.is_a if hasattr(p, 'name') and p.name != 'Thing']
            
            # Determine parent index (-1 if no parent or root level)
            parent_index = -1
            if parent_classes:
                parent_name = parent_classes[0].name  # Take first parent
                parent_index = class_to_index.get(parent_name, -1)
            
            # Get summary from comment, summary property, or hasText property
            summary = ""
            if hasattr(cls, 'comment') and cls.comment:
                summary = cls.comment[0] if isinstance(cls.comment, list) else str(cls.comment)
            elif hasattr(cls, 'summary') and cls.summary:
                summary = cls.summary[0] if isinstance(cls.summary, list) else str(cls.summary)
            elif hasattr(cls, 'hasText') and cls.hasText:
                text = cls.hasText[0] if isinstance(cls.hasText, list) else str(cls.hasText)
                # Truncate long text for summary
                summary = text[:200] + "..." if len(text) > 200 else text
            
            node = {
                "index": class_to_index[cls.name],
                "parent_index": parent_index,
                "name": cls.name,
                "summary": summary
            }
            
            nodes.append(node)
        
        # Sort by index to maintain consistent ordering
        nodes.sort(key=lambda x: x["index"])
        
        return {"nodes": nodes}
        
    except Exception as e:
        print(f"Error extracting ontology info: {e}")
        return None

# Add these new API endpoints to your existing Flask app

@app.route('/generate-ontology', methods=['POST'])
def generate_ontology():
    """Generate ontology from JSON structure"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Initialize the ontology generator with existing client
        generator = OntologyMindmapGenerator(client)
        
        # Generate a unique filename
        import uuid
        filename = f"ontology_{uuid.uuid4().hex[:8]}"
        
        # Process the JSON data and create ontology
        output_path, message = generator.process_json_data(data, filename)
        
        if not output_path:
            return jsonify({"error": message}), 422
        
        return jsonify({
            "message": message,
            "ontology_file": output_path,
            "filename": f"{filename}.owl"
        }), 200
        
    except Exception as e:
        print(f"Error generating ontology: {e}")
        return jsonify({"error": f"Failed to generate ontology: {str(e)}"}), 500

@app.route('/ontology-info/<filename>', methods=['GET'])
def get_ontology_info(filename):
    """Get information from OWL file in JSON format"""
    try:
        # Construct the full path to the ontology file
        ontology_path = os.path.join('static', filename)
        
        if not os.path.exists(ontology_path):
            return jsonify({"error": "Ontology file not found"}), 404
        
        # Extract information from the ontology
        ontology_data = extract_ontology_info(ontology_path)
        
        if not ontology_data:
            return jsonify({"error": "Failed to extract ontology information"}), 500
        
        return jsonify(ontology_data), 200
        
    except Exception as e:
        print(f"Error getting ontology info: {e}")
        return jsonify({"error": f"Failed to get ontology information: {str(e)}"}), 500

@app.route('/list-ontologies', methods=['GET'])
def list_ontologies():
    """List all available ontology files"""
    try:
        static_dir = 'static'
        if not os.path.exists(static_dir):
            return jsonify({"ontologies": []}), 200
        
        ontology_files = [f for f in os.listdir(static_dir) if f.endswith('.owl')]
        
        ontologies_info = []
        for filename in ontology_files:
            file_path = os.path.join(static_dir, filename)
            file_stats = os.stat(file_path)
            
            ontologies_info.append({
                "filename": filename,
                "created": file_stats.st_ctime,
                "size": file_stats.st_size
            })
        
        return jsonify({"ontologies": ontologies_info}), 200
        
    except Exception as e:
        print(f"Error listing ontologies: {e}")
        return jsonify({"error": f"Failed to list ontologies: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    data = request.json
    question = data.get("message", "")
    
    start_time = time.time()
    relation = find_relation(onto)
    print("\n====================== ENTITIES ==============================")
    entities_with_annotation_sumarry = get_entities_with_annotation(onto, 'summary')
    explication = create_explication(entities_with_annotation_sumarry)

    # Convert chat_histories format for LLM (role, content)
    llm_chat_history = [{"role": "user" if h["sender"] == "user" else "assistant", "content": h["text"]} for h in chat_histories[user_id]]

    entities = find_entities_from_question_PP1(relation, explication, question, llm_chat_history)

    list_query = []
    try:
        parsed_entities = json.loads(entities)
        list_query = create_query(name_ontology, parsed_entities)
        print("list_query: ", list_query)
    except json.JSONDecodeError as e:
        print(f"Error decoding entities JSON from LLM: {e}. Raw LLM response: {entities}")
        return jsonify({"response": "Xin lỗi, tôi gặp vấn đề khi xử lý yêu cầu của bạn. Vui lòng thử lại."})
    except Exception as e:
        print(f"Error creating query: {e}")
        return jsonify({"response": "Xin lỗi, tôi gặp vấn đề khi tạo truy vấn từ thực thể. Vui lòng thử lại."})


    print("\n====================== KẾT QUẢ TRA CỨU =======================\n")
    result_from_ontology = find_question_info(name_ontology, list_query)
    print("result_from_ontology: ", result_from_ontology)

    raw_informations_from_ontology = []
    if result_from_ontology:
        for result_list in result_from_ontology: # result_from_ontology is list of lists
            raw_informations_from_ontology.extend([item for item in result_list if item is not None])
    else:
        raw_informations_from_ontology.append("Không có thông tin cho câu hỏi")

    k_similar_info = find_similar_info_from_raw_informations(question, raw_informations_from_ontology)
    print("\n====================== SIMILAR INFO ===========================\n")
    print(k_similar_info)

    # Pass chat_histories format for LLM (role, content)
    bot_response = generate_response(relation, k_similar_info, question, llm_chat_history)
    end_time = time.time()
    print("Thời gian thực thi:", end_time - start_time, "giây")

    chat_histories[user_id].append({"sender": "user", "text": question})
    chat_histories[user_id].append({"sender": "bot", "text": bot_response})

    return jsonify({"response": bot_response})


@app.route('/')
def index():
    return "Backend is running. Send POST requests to /upload for document processing or /chat for chatbot interaction."

if __name__ == "__main__":
    # Create the static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    # Run the Flask app
    app.run(debug=True, port=5000)
