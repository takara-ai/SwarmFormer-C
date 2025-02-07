#ifndef SWARMFORMER_CONFIG_H
#define SWARMFORMER_CONFIG_H

#include <stdint.h>

typedef struct {
    char* host;
    uint16_t port;
    char* weights_path;
} Config;

Config* load_config(const char* config_path);
void free_config(Config* config);

#endif 