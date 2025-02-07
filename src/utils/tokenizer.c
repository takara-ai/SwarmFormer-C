#include "tokenizer.h"

static void to_lower(char* text) {
    for (char* p = text; *p; p++) {
        *p = tolower(*p);
    }
}

static int is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int is_punctuation(char c) {
    return (c >= 33 && c <= 47) ||
           (c >= 58 && c <= 64) ||
           (c >= 91 && c <= 96) ||
           (c >= 123 && c <= 126);
}

static int find_token(Tokenizer* tokenizer, const char* token) {
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        if (strcmp(token, tokenizer->tokens[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static int find_longest_subword(Tokenizer* tokenizer, const char* word, int start, int len, char* buffer) {
    int max_len = len - start;
    if (max_len <= 0) return -1;
    
    for (int subword_len = max_len; subword_len > 0; subword_len--) {
        strncpy(buffer, word + start, subword_len);
        buffer[subword_len] = '\0';
        
        char temp[MAX_TOKEN_LENGTH];
        const char* search_token;
        if (start > 0) {
            snprintf(temp, sizeof(temp), "##%s", buffer);
            search_token = temp;
        } else {
            search_token = buffer;
        }
        
        int token_id = find_token(tokenizer, search_token);
        if (token_id >= 0) {
            return token_id;
        }
    }
    
    return -1;
}

Tokenizer* load_tokenizer(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "rb");
    if (!f) {
        printf("Error: Could not open vocabulary file %s\n", vocab_path);
        return NULL;
    }
    
    Tokenizer* tokenizer = (Tokenizer*)malloc(sizeof(Tokenizer));
    fread(&tokenizer->vocab_size, sizeof(int), 1, f);
    fread(&tokenizer->max_token_length, sizeof(int), 1, f);
    tokenizer->tokens = (char**)malloc(tokenizer->vocab_size * sizeof(char*));

    for (int i = 0; i < tokenizer->vocab_size; i++) {
        int token_length;
        fread(&token_length, sizeof(int), 1, f);
        tokenizer->tokens[i] = (char*)malloc((token_length + 1) * sizeof(char));
        fread(tokenizer->tokens[i], sizeof(char), token_length, f);
        tokenizer->tokens[i][token_length] = '\0';
    }
    
    fclose(f);
    return tokenizer;
}

void free_tokenizer(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->tokens[i]);
    }
    free(tokenizer->tokens);
    free(tokenizer);
}

int* tokenize_text(Tokenizer* tokenizer, const char* text, int max_length, int* num_tokens) {
    int* tokens = (int*)malloc(max_length * sizeof(int));
    *num_tokens = 0;
    
    char* text_copy = strdup(text);
    to_lower(text_copy);
    
    char buffer[MAX_TOKEN_LENGTH];
    char word[MAX_TOKEN_LENGTH];
    int word_start = 0;
    int word_len = 0;
    
    for (int i = 0; text_copy[i] != '\0' && *num_tokens < max_length - 2; i++) {
        char c = text_copy[i];
        
        if (is_whitespace(c) || is_punctuation(c) || text_copy[i + 1] == '\0') {
            if (word_len > 0 || (!is_whitespace(c) && text_copy[i + 1] == '\0')) {
                if (text_copy[i + 1] == '\0' && !is_whitespace(c)) {
                    word[word_len++] = c;
                }
                
                strncpy(word, text_copy + word_start, word_len);
                word[word_len] = '\0';
                
                int token_id = find_token(tokenizer, word);
                if (token_id >= 0) {
                    tokens[(*num_tokens)++] = token_id;
                } else {
                    int pos = 0;
                    while (pos < word_len && *num_tokens < max_length - 2) {
                        token_id = find_longest_subword(tokenizer, word, pos, word_len, buffer);
                        if (token_id >= 0) {
                            tokens[(*num_tokens)++] = token_id;
                            pos += strlen(buffer);
                        } else {
                            tokens[(*num_tokens)++] = UNK_TOKEN_ID;
                            break;
                        }
                    }
                }
                
                word_len = 0;
            }
            
            if (is_punctuation(c)) {
                buffer[0] = c;
                buffer[1] = '\0';
                int token_id = find_token(tokenizer, buffer);
                if (token_id >= 0) {
                    tokens[(*num_tokens)++] = token_id;
                }
            }
            
            word_start = i + 1;
        } else {
            word_len++;
        }
    }
    
    free(text_copy);
    
    add_special_tokens(tokens, num_tokens, max_length);
    
    return tokens;
}

void add_special_tokens(int* tokens, int* num_tokens, int max_length) {
    if (*num_tokens >= max_length) {
        *num_tokens = max_length - 2;  // Make room for CLS and SEP tokens
    }
    
    memmove(tokens + 1, tokens, (*num_tokens) * sizeof(int));
    tokens[0] = CLS_TOKEN_ID;
    
    if (*num_tokens < max_length - 1) {
        tokens[*num_tokens + 1] = SEP_TOKEN_ID;
        *num_tokens += 2;
    }
    
    while (*num_tokens < max_length) {
        tokens[*num_tokens] = PAD_TOKEN_ID;
        (*num_tokens)++;
    }
} 