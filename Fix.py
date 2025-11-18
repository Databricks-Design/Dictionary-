if use_memory_zone:
            self.logger.log_info(get_log(logging.INFO, "Executing with memory_zone."))
            
            # 1. Run Pipeline (Existing code)
            with self.nlp.memory_zone():
                result_docs = self.nlp.pipe(input_all)
                _, ret_labels, ret_extracted_texts, ret_ent_ids = self.extract_results(result_docs, descriptions_all, memos_all)

            # ==================================================================
            # 2. PASTE THIS BLOCK HERE (The Soft Reset)
            # ==================================================================
            if hasattr(self.nlp, "tokenizer"):
                 from spacy.tokenizer import Tokenizer
                 # Replace the tokenizer with a fresh one to kill the internal cache
                 self.nlp.tokenizer = Tokenizer(
                     self.nlp.vocab,
                     self.nlp.tokenizer.rules,
                     self.nlp.tokenizer.prefix_search,
                     self.nlp.tokenizer.suffix_search,
                     self.nlp.tokenizer.infix_finditer,
                     self.nlp.tokenizer.token_match,
                     self.nlp.tokenizer.url_match,
                 )
            # ==================================================================

        else:
            # ... (rest of your code)
