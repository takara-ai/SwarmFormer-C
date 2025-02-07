#ifndef SWARMFORMER_HTTP_SERVER_H
#define SWARMFORMER_HTTP_SERVER_H

#include <stdint.h>
#include "../model/swarmformer.h"
#include "../utils/tokenizer.h"

typedef struct {
    const char* host;
    uint16_t port;
    SwarmFormerModel* model;
    Tokenizer* tokenizer;
} ServerConfig;

int start_server(ServerConfig* config);

#endif 