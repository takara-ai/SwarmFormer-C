#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

#define MAX_LINE_LENGTH 1024

static char* trim(char* str) {
    while(*str == ' ' || *str == '\t') str++;
    char* end = str + strlen(str) - 1;
    while(end > str && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end--;
    *(end + 1) = '\0';
    return str;
}

Config* load_config(const char* config_path) {
    FILE* file = fopen(config_path, "r");
    if (!file) {
        printf("Error: Could not open config file %s\n", config_path);
        return NULL;
    }

    Config* config = (Config*)calloc(1, sizeof(Config));
    config->host = strdup("127.0.0.1");
    config->port = 8080;

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        char* key = strtok(line, "=");
        char* value = strtok(NULL, "=");
        
        if (!key || !value) continue;
        
        key = trim(key);
        value = trim(value);
        
        if (strcmp(key, "host") == 0) {
            free(config->host);
            config->host = strdup(value);
        } else if (strcmp(key, "port") == 0) {
            config->port = (uint16_t)atoi(value);
        } else if (strcmp(key, "weights_path") == 0) {
            config->weights_path = strdup(value);
        }
    }

    fclose(file);
    return config;
}

void free_config(Config* config) {
    if (config) {
        free(config->host);
        free(config->weights_path);
        free(config);
    }
} 