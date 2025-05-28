class OntologyMindmapGenerator:
    def __init__(self, client: OpenAI, model_embedding_path: str = None):
        self.client = client
        self.onto = None # This will hold the current ontology object
        # Initialize SentenceTransformer model if path is provided
        self.model_embedding = None
        if model_embedding_path:
            try:
                self.model_embedding = SentenceTransformer(model_embedding_path)
                print(f"SentenceTransformer model loaded from: {model_embedding_path}")
            except Exception as e:
                print(f"Error loading SentenceTransformer model from {model_embedding_path}: {e}")
                print("Proceeding without sentence embeddings for similarity. This might affect 'find_max_similarity_score_of_list_with_title'.")
        else:
            print("No path provided for SentenceTransformer model. Sentence embeddings will not be used for similarity.")

        # For flatten_json (equivalent to global current_index in Colab)
        self._current_flatten_index = 0

    # --- Utility Functions (moved from global scope in Colab) ---

    def _get_embedding(self, text, model="text-embedding-ada-002"):
        if not text:
            return [] # Return empty list for empty text
        if self.model_embedding:
            try:
                return self.model_embedding.encode(text).tolist() # Convert numpy array to list
            except Exception as e:
                print(f"Error encoding text with SentenceTransformer: {e}. Falling back to OpenAI if available.")
        
        # Fallback to OpenAI if SentenceTransformer failed or not available
        if self.client:
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=model
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error getting embedding from OpenAI: {e}")
                return [] # Return empty list on failure
        return []

    def _summary_paragraph(self, paragraph):
        system_prompt = '''
            Bạn là chuyên giao trong việc tóm tắt ngắn gọn các văn bản lịch sử.
            Hãy tóm tắt ngắn gọn đoạn văn được cung cấp nhưng tuyệt đối không được làm mất đi các thông tin lịch sử quan trọng.
            '''
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paragraph}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error summarizing paragraph with OpenAI: {e}")
            return paragraph # Return original if summarization fails

    def _extract_key_word(self, summary):
        system_prompt = '''
            Bạn là chuyên gia trong việc trích xuất từ khóa cho thông tin lịch sử.
            Hãy tìm ra một từ/cụm từ khóa có thể thể hiện tổng quát nội dung cốt lõi của đoạn văn.
            YÊU CẦU:
            Chỉ cung cấp từ khóa, không đưa thông tin gì thêm.
            '''
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": summary}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error extracting keyword with OpenAI: {e}")
            return "" # Return empty string on failure

    def _compute_similarity_score(self, title, paragraph):
        e_title = self._get_embedding(title)
        e_paragraph = self._get_embedding(paragraph)

        if not e_title or not e_paragraph:
            return 0.0 # Cannot compute similarity if embeddings are missing

        # Ensure both are 2D arrays for cosine_similarity
        return cosine_similarity([e_title], [e_paragraph])[0][0]

    # --- Flattening and Merging Functions ---

    def _flatten_json(self, node, parent_index=None, flattened_list=None):
        if flattened_list is None:
            flattened_list = []
        
        if isinstance(node, dict):
            for key, value in node.items():
                if key != 'null': # Filter out 'null' keys
                    my_index = self._current_flatten_index
                    flattened_list.append({
                        "text": key,
                        "chosen_summary": key, # Initial state, will be updated later
                        "index": my_index,
                        "index_parent": parent_index
                    })
                    self._current_flatten_index += 1
                    self._flatten_json(value, my_index, flattened_list)
                else: # Handle null keys - still recurse on their values
                    self._flatten_json(value, parent_index, flattened_list)
        elif isinstance(node, list):
            for item in node:
                self._flatten_json(item, parent_index, flattened_list)
        elif isinstance(node, str):
            if node != 'null': # Filter out 'null' strings
                flattened_list.append({
                    "text": node,
                    "chosen_summary": node, # Initial state, will be updated later
                    "index": self._current_flatten_index,
                    "index_parent": parent_index
                })
                self._current_flatten_index += 1
        return flattened_list

    def _merge_short_nodes(self, flattened, word_limit=100):
        # Initialize chosen_summary structure for all items
        for item in flattened:
            item["chosen_summary"] = {
                "full_text": item["text"],
                "summary": "",
                "keyword": ""
            }

        leaf_nodes_indices = set(item["index"] for item in flattened)
        parent_indices = set(item["index_parent"] for item in flattened if item["index_parent"] is not None)
        leaf_nodes_only = leaf_nodes_indices - parent_indices

        nodes_by_parent = {}
        for i, item in enumerate(flattened):
            parent = item["index_parent"]
            if parent not in nodes_by_parent:
                nodes_by_parent[parent] = []
            nodes_by_parent[parent].append(i)

        result_flattened = flattened.copy()
        nodes_to_remove = set()

        for parent, indices in nodes_by_parent.items():
            # Sort indices to process in document order
            indices.sort(key=lambda idx: result_flattened[idx]["index"]) # Sort by original index

            # Filter for actual leaf nodes within this parent's children
            current_parent_leaf_indices = [
                idx for idx in indices if result_flattened[idx]["index"] in leaf_nodes_only
            ]

            if not current_parent_leaf_indices:
                continue # No leaf nodes for this parent

            # Iterate through the leaf nodes from left to right (document order)
            for i in range(len(current_parent_leaf_indices) - 1, 0, -1): # Iterate backwards
                current_idx_in_flat = current_parent_leaf_indices[i]
                prev_idx_in_flat = current_parent_leaf_indices[i-1]

                current_node = result_flattened[current_idx_in_flat]
                prev_node = result_flattened[prev_idx_in_flat]

                if current_node["index_parent"] == prev_node["index_parent"] and \
                   len(current_node["text"].split()) < word_limit:
                    
                    # Merge current_node into prev_node
                    prev_node["text"] += " " + current_node["text"]
                    prev_node["chosen_summary"]["full_text"] += " " + current_node["chosen_summary"]["full_text"]
                    
                    nodes_to_remove.add(current_idx_in_flat)

        final_result = [result_flattened[i] for i in range(len(result_flattened)) if i not in nodes_to_remove]

        # Re-index remaining nodes sequentially from 0
        old_to_new_index_map = {}
        for i, item in enumerate(final_result):
            old_to_new_index_map[item["index"]] = i
            item["index"] = i

        # Update parent indices using the new mapping
        for item in final_result:
            if item["index_parent"] is not None:
                item["index_parent"] = old_to_new_index_map.get(item["index_parent"], None)
                # If a parent was removed/merged, its children become top-level or are handled by their new parent.
                # Setting to 0 (root) if parent was removed or not found in new map.
                if item["index_parent"] is None:
                    item["index_parent"] = 0 # Assign to a default root index if parent disappeared
        
        # Ensure that nodes whose original parent was 0 (root) still reflect that.
        # This part might need careful consideration if the original 0 refers to a conceptual root.
        # If the root is a specific "Lịch_sử_Việt_Nam" class that isn't merged, its index needs to be handled.
        # For simplicity, if the new mapping yields None for a parent, assign 0.
        
        return final_result


    # --- Bottom-Up Processing Functions ---

    def _build_tree_for_bottom_up(self, flattened_list):
        children_map = defaultdict(list)
        index_map = {}
        for item in flattened_list:
            index_map[item['index']] = item
            children_map[item['index_parent']].append(item) # Parent_index can be None or 0 (root)
        return children_map, index_map

    def _get_text_by_index(self, flattened_list, index):
        # Assuming index is unique and exists in flattened_list
        text_content = [item['text'] for item in flattened_list if item['index'] == index]
        return text_content[0] if text_content else ""


    def _find_max_similarity_score_of_list_with_title(self, title_parent, list_contents):
        max_score = -1.0 # Cosine similarity ranges from -1 to 1
        max_score_index = -1
        
        if not list_contents:
            return -1, 0.0

        for idx, content in enumerate(list_contents):
            # Only compute if content is not empty
            if content.strip():
                similar_score = self._compute_similarity_score(title_parent, content)
                if similar_score > max_score:
                    max_score = similar_score
                    max_score_index = idx
        return max_score_index, max_score

    def _bottom_up_replace_summary(self, flattened_list):
        # This function modifies the list in place (via index_map)
        children_map, index_map = self._build_tree_for_bottom_up(flattened_list)

        # Get list of all indices and sort them in reverse order (bottom-up)
        sorted_indexes = sorted(index_map.keys(), reverse=True)

        for idx in sorted_indexes:
            current_node_data = index_map[idx]
            
            # Check if current node has children (is not a leaf in this iteration)
            if idx in children_map and children_map[idx]:
                children = children_map[idx]
                
                # Get full_text from children's chosen_summary
                list_child_texts = [child['chosen_summary']['full_text'] for child in children if child['chosen_summary']]

                # If there are no meaningful child texts, skip this node or use its own text
                if not list_child_texts or all(not text.strip() for text in list_child_texts):
                    # If children are empty, consider this node a leaf or just use its own text
                    full_text = current_node_data['text']
                    if full_text and full_text != "null":
                        summarized_paragraph = self._summary_paragraph(full_text)
                        keyword = self._extract_key_word(summarized_paragraph)
                        current_node_data['chosen_summary'] = {
                            "full_text": full_text,
                            "summary": summarized_paragraph,
                            "keyword": keyword
                        }
                    continue

                parent_title = self._get_text_by_index(flattened_list, idx)
                chosen_summary_idx, _ = self._find_max_similarity_score_of_list_with_title(parent_title, list_child_texts)

                if chosen_summary_idx != -1:
                    chosen_summary_full_text = list_child_texts[chosen_summary_idx]
                    summarized_paragraph = self._summary_paragraph(chosen_summary_full_text)
                    keyword = self._extract_key_word(summarized_paragraph)

                    current_node_data['chosen_summary'] = {
                        "full_text": chosen_summary_full_text,
                        "summary": summarized_paragraph,
                        "keyword": keyword
                    }
                else:
                    # Fallback if no suitable child summary is found
                    full_text = current_node_data['text']
                    if full_text and full_text != "null":
                        summarized_paragraph = self._summary_paragraph(full_text)
                        keyword = self._extract_key_word(summarized_paragraph)
                        current_node_data['chosen_summary'] = {
                            "full_text": full_text,
                            "summary": summarized_paragraph,
                            "keyword": keyword
                        }
            else: # Node is a true leaf (has no children)
                full_text = current_node_data['text']
                if full_text and full_text != "null":
                    summarized_paragraph = self._summary_paragraph(full_text)
                    keyword = self._extract_key_word(summarized_paragraph)
                    current_node_data['chosen_summary'] = {
                        "full_text": full_text,
                        "summary": summarized_paragraph,
                        "keyword": keyword
                    }
        # The modifications happened in index_map, which references the original objects in flattened_list.
        # So flattened_list itself is updated.
        return flattened_list

    # --- Ontology Creation Functions ---

    def _safe_add_annotation_property(self, onto, annotation_name):
        with onto:
            # Check if property already exists in the current ontology or globally
            if not hasattr(onto, annotation_name):
                # Using types.new_class to create annotation property
                new_prop = types.new_class(annotation_name, (AnnotationProperty,))
                setattr(onto, annotation_name, new_prop) # Attach to ontology object directly
                print(f"Created annotation property: {annotation_name}")
            else:
                print(f"Annotation property '{annotation_name}' already exists.")


    def _group_nodes_by_parent(self, merged_nodes_list):
        nodes_by_parent = defaultdict(list)
        for node in merged_nodes_list:
            parent_index = node["index_parent"]
            nodes_by_parent[parent_index].append(node)
        return nodes_by_parent

    def _clean_class_name(self, name):
        name = name.strip()
        name = re.sub(r'[^\w\s-]', '', name) # Remove special characters except alphanumeric, whitespace, hyphen
        name = name.replace(' ', '_') # Replace spaces with underscores
        name = name[:200] # Limit length

        # Ensure name starts with a letter or underscore
        if name and not (name[0].isalpha() or name[0] == '_'):
            name = '_' + name
        
        if not name:
            name = 'UnnamedClass'
        return name

    def _add_class_to_ontology(self, onto, parent_class_name, class_name_list, node, all_merged_nodes):
        class_name = class_name_list[0] # Get current name from list
        with onto:
            parent_class = getattr(onto, parent_class_name, None)
            if not parent_class:
                # If parent class somehow doesn't exist, make it a top-level Thing
                # This should ideally not happen if process_nodes_level_by_level is structured correctly
                parent_class = types.new_class(parent_class_name, (Thing,))
                print(f"Warning: Parent class '{parent_class_name}' not found, creating as Thing.")


            # Handle duplicate class names in OWLready2
            if hasattr(onto, class_name): # Check if the name is already taken in the ontology
                i = 1
                while True:
                    new_class_name = f"{class_name}_{'1'*i}"
                    if not hasattr(onto, new_class_name):
                        class_name_list[0] = new_class_name # Update the list with the new name
                        break
                    i += 1

            new_class = types.new_class(class_name_list[0], (parent_class,))

            # Ensure annotation properties exist in the current ontology context
            self._safe_add_annotation_property(onto, "summary")
            self._safe_add_annotation_property(onto, "summary_embeddings")

            # Find the corresponding node in all_merged_nodes to get its summary
            # This is already the 'node' parameter passed in, so we don't need to loop.
            summary_value = node["chosen_summary"]["summary"]
            full_text_value = node["chosen_summary"]["full_text"] # Keep full text for potential debugging/context

            if summary_value:
                new_class.summary = summary_value
                # Encode the summary for embedding
                summary_embedding = self._get_embedding(summary_value)
                if summary_embedding: # Only set if embedding was successfully generated
                    new_class.summary_embeddings = json.dumps(summary_embedding) # Store as JSON string

            # Add rdfs:label for better display in Protege
            new_class.label.append(node["Node_name"]) # Use the human-readable Node_name from convert_ontology_to_json
            # new_class.comment.append(full_text_value) # Optional: add full text as a comment

    def _process_nodes_level_by_level(self, onto, nodes_by_parent, class_names_map, all_merged_nodes, parent_index=None):
        current_level_nodes = nodes_by_parent.get(parent_index, [])
        
        # Sort nodes at the current level by their original index to maintain document order
        current_level_nodes.sort(key=lambda n: n["index"])

        for node in current_level_nodes:
            index = node["index"]
            # Use a mutable list to allow add_class_to_ontology to update the name
            current_class_name_list = [class_names_map[index]] 

            # Determine parent class name. If parent_index is None, it's a root concept.
            # We explicitly link to "Lịch_sử_Việt_Nam" if parent_index is None
            parent_class_owl_name = "Lịch_sử_Việt_Nam" if parent_index is None else class_names_map[parent_index]
            
            # Add class and its annotations/relationships
            self._add_class_to_ontology(onto, parent_class_owl_name, current_class_name_list, node, all_merged_nodes)
            
            # Update the class_names_map with the potentially modified class name
            class_names_map[index] = current_class_name_list[0]

            # Recursively process children
            self._process_nodes_level_by_level(onto, nodes_by_parent, class_names_map, all_merged_nodes, index)


    def _create_ontology_structure(self, merged_nodes_list, ontology_iri="http://www.semanticweb.org/MINDMAP"):
        self.onto = get_ontology(ontology_iri) # Initialize ontology object
        
        # Dictionary to store clean class names based on their original index
        class_names_map = {}

        # Step 1: Pre-process nodes to determine cleaned class names and their summaries/keywords
        # This is where Node_name is derived from keyword or text.
        for node in merged_nodes_list:
            index = node["index"]
            keyword = node["chosen_summary"]["keyword"]
            
            # Use Node_name as the cleaner name for display in JSON output.
            # Use cleaned_keyword as the OWL class name.
            cleaned_keyword = self._clean_class_name(keyword if keyword else node["text"])
            
            # Update node's Node_name and set cleaned_keyword as its potential OWL class name
            node["Node_name"] = keyword if keyword else node["text"] # Store the human-readable name
            class_names_map[index] = cleaned_keyword # Store the OWL-compatible name for mapping

        # Step 2: Create the root class "Lịch_sử_Việt_Nam" and annotation properties
        with self.onto:
            root_class = types.new_class("Lịch_sử_Việt_Nam", (Thing,))
            self._safe_add_annotation_property(self.onto, "summary")
            self._safe_add_annotation_property(self.onto, "summary_embeddings")

        # Step 3: Group nodes by parent for hierarchical processing
        nodes_by_parent = self._group_nodes_by_parent(merged_nodes_list)
        
        # Step 4: Recursively process and add classes to the ontology
        # Start with nodes that have parent_index 0 (or None, which maps to root)
        # Note: Your Colab logic uses index_parent=0 for root.
        # We'll pass None as the initial parent_index to link to the conceptual root.
        # Ensure that nodes with index_parent 0 are processed as children of "Lịch_sử_Việt_Nam"
        self._process_nodes_level_by_level(self.onto, nodes_by_parent, class_names_map, merged_nodes_list, parent_index=0) # Changed to 0 based on your `flatten_json`'s `index_parent` for root

        return self.onto


    # --- Main Processing Method for OntologyMindmapGenerator ---
    def process_json_data(self, input_json_data, output_prefix="ontology_output"):
        """
        Processes the input hierarchical JSON data to generate an OWL ontology
        and a flattened JSON representation suitable for the frontend.

        Args:
            input_json_data (dict): The structured JSON data (tree_json) from TextAnalyzer.
            output_prefix (str): Prefix for the output OWL and JSON files.

        Returns:
            tuple: (owl_file_path, json_ontology_path) or (None, None) on error.
        """
        print("Starting ontology generation process...")
        try:
            # 1. Flatten the input JSON
            self._current_flatten_index = 1 # Reset index for each new processing (as per your example starting index from 1)
            flattened_data = self._flatten_json(input_json_data)
            print(f"Flattened data: {len(flattened_data)} nodes.")
            # print("Flattened Data Sample:", json.dumps(flattened_data[:5], ensure_ascii=False, indent=2))


            # 2. Merge short nodes
            # Assuming word_limit=100 from your Colab notebook
            merged_nodes = self._merge_short_nodes(flattened_data, word_limit=100)
            print(f"Merged nodes: {len(merged_nodes)} nodes after merging.")
            # print("Merged Nodes Sample:", json.dumps(merged_nodes[:5], ensure_ascii=False, indent=2))


            # 3. Apply bottom-up summarization and keyword extraction
            # This modifies merged_nodes in-place
            final_processed_nodes = self._bottom_up_replace_summary(merged_nodes)
            print("Completed bottom-up summarization and keyword extraction.")
            # print("Final Processed Nodes Sample:", json.dumps(final_processed_nodes[:5], ensure_ascii=False, indent=2))

            # 4. Create OWL Ontology
            # Ensure the output directory exists
            output_dir = "static"
            os.makedirs(output_dir, exist_ok=True)
            owl_file_name = f"{output_prefix}.owl"
            owl_file_path = os.path.join(output_dir, owl_file_name)
            
            self.onto = self._create_ontology_structure(final_processed_nodes)
            self.onto.save(file=owl_file_path, format="rdfxml")
            print(f"Ontology saved to: {owl_file_path}")

            # 5. Convert Ontology to Frontend-friendly JSON (your desired format)
            # This uses the specific format from the first image you provided
            frontend_json_output = self._convert_ontology_to_json_desired_format(final_processed_nodes)

            json_ontology_file_name = f"{output_prefix}.json"
            json_ontology_path = os.path.join(output_dir, json_ontology_file_name)
            with open(json_ontology_path, 'w', encoding='utf-8') as f:
                json.dump(frontend_json_output, f, ensure_ascii=False, indent=4)
            print(f"Frontend JSON ontology saved to: {json_ontology_path}")

            return owl_file_path, json_ontology_path

        except Exception as e:
            print(f"Error during ontology generation process: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    # --- New `convert_ontology_to_json` to match your desired format ---
    # This is distinct from your old convert_ontology_to_json (which produced nodes/edges)
    # This one will take the `final_processed_nodes` directly because it already has the
    # Index, Node_name, summary, and Parent_index structure.
    def _convert_ontology_to_json_desired_format(self, final_processed_nodes):
        """
        Converts the processed list of nodes into the specific JSON format
        (list of dicts with Index, Node_name, summary, Parent_index).
        This directly uses the structure prepared by bottom_up_replace_summary.
        """
        # The final_processed_nodes list *already* contains the structure you want,
        # provided that the 'Node_name' and 'summary' fields are set correctly
        # during the _bottom_up_replace_summary and _create_ontology_structure steps.
        
        desired_output = []
        for node in final_processed_nodes:
            # Ensure the Node_name is correctly set (it's often the keyword/text)
            # And summary is from chosen_summary
            
            # The 'Node_name' should ideally be the `keyword` if present, else the `text`
            # and it should be clean. Let's make sure it's set consistently.
            # In `_create_ontology_structure`, I've already ensured `node["Node_name"]`
            # is set to the human-readable keyword/text.
            
            desired_output.append({
                "Index": node["index"],
                "Node_name": node.get("Node_name", node["text"]), # Fallback to text if Node_name not explicitly set
                "summary": node["chosen_summary"]["summary"],
                "Parent_index": node["index_parent"] if node["index_parent"] is not None else 0
            })
        
        # Sort by index as in your example output
        desired_output.sort(key=lambda x: x['Index'])
        return desired_output
    
ontology_generator = OntologyMindmapGenerator(client=client)

@app.route('/process-document', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type"}), 400

    start_time = time.time()
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)

    try:
        pdf_processor = PDFProcessor(client)
        text_analyzer = TextAnalyzer(client)
        
        paragraphs = []
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension == 'pdf':
            # For PDF, you might want to dynamically get start/end pages from request if needed
            paragraphs = pdf_processor.extract_text_from_pdf(file_path, 1, 9999) # Process all pages
        elif file_extension == 'docx':
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                paragraphs = [line for line in f.read().splitlines() if line.strip()]
        elif file_extension in ['png', 'jpg', 'jpeg']:
            # Perform OCR on image files
            img = Image.open(BytesIO(file.read()))
            ocr_text = pytesseract.image_to_string(img, lang='vie') # Assuming 'vie' for Vietnamese
            paragraphs = [line for line in ocr_text.splitlines() if line.strip()]
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        processed_paragraphs = pdf_processor.preprocess_paragraphs(paragraphs)
        df = text_analyzer.create_dataframe(processed_paragraphs)
        tree_json = text_analyzer.build_tree(df)

        end_time_processing = time.time()
        processing_time = end_time_processing - start_time

        # --- NEW: Trigger Ontology Generation and Conversion ---
        owl_file_path, json_ontology_path = ontology_generator.process_json_data(
            tree_json,
            output_prefix=f"{os.path.splitext(filename)[0]}_ontology" # Ontology name based on filename
        )
        # --- END NEW ---

        end_time_total = time.time()
        total_execution_time = end_time_total - start_time


        return jsonify({
            "message": "Document processed and ontology generated successfully!",
            "structured_data": tree_json,
            "processing_time_seconds": processing_time,
            "ontology_owl_file": owl_file_path,
            "ontology_json_file": json_ontology_path,
            "total_execution_time_seconds": total_execution_time
        }), 200

    except Exception as e:
        print(f"Error processing document: {e}")
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.route('/generate-ontology-from-json', methods=['POST'])
def generate_ontology_from_json():
    start_time = time.time()
    
    # Check if the request body is JSON and contains the 'data' field
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    request_data = request.get_json()
    if 'data' not in request_data:
        return jsonify({"error": "No 'data' field found in the JSON request body. Please send your structured JSON under a 'data' key."}), 400

    processed_json_data = request_data['data']

    if not processed_json_data:
        return jsonify({"message": "No structured JSON data provided for ontology generation."}), 200

    try:
        # Assuming you want a default output prefix, or you could get it from the request
        # For a more specific name, you might need to infer it or have the frontend send it.
        # Let's use a generic name or generate one based on timestamp/random string.
        output_prefix_name = f"ontology_{int(time.time())}" # Unique name based on timestamp

        owl_file_path, json_ontology_path = ontology_generator.process_json_data(
            processed_json_data,
            output_prefix=output_prefix_name
        )

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Ontology generation and conversion from JSON took {execution_time:.2f} seconds.")

        return jsonify({
            "message": "Ontology and JSON generated successfully from provided JSON data!",
            "owl_file": owl_file_path,
            "json_file": json_ontology_path,
            "execution_time": execution_time
        }), 200

    except Exception as e:
        print(f"Error during ontology generation from JSON: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc() 
        return jsonify({"error": f"Failed to generate ontology from JSON: {str(e)}"}), 500


# --- Root endpoint (optional, just for basic check) ---
@app.route('/')
def index():
    return "Backend is running. Send POST requests to /upload for document processing or /chat for chatbot interaction."

if __name__ == "__main__":
    # Create the static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    # Run the Flask app
    app.run(debug=True, port=5000)
