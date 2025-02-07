#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define VOCAB_SIZE 30522
#define MAX_TOKEN_LENGTH 64
#define PAD_TOKEN_ID 0
#define UNK_TOKEN_ID 100
#define CLS_TOKEN_ID 101
#define SEP_TOKEN_ID 102
#define MASK_TOKEN_ID 103

typedef struct {
    char** tokens;
    int vocab_size;
    int max_token_length;
} Tokenizer;

Tokenizer* load_tokenizer(const char* vocab_path);
void free_tokenizer(Tokenizer* tokenizer);

int* tokenize_text(Tokenizer* tokenizer, const char* text, int max_length, int* num_tokens);
void add_special_tokens(int* tokens, int* num_tokens, int max_length);

#endif