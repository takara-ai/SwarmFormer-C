#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
typedef int socklen_t;
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
typedef int socket_t;
#define CLOSE_SOCKET(s) close(s)
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#endif

#include "http_server.h"

#define BUFFER_SIZE 4096
#define MAX_HEADERS 100
#define MAX_BODY_SIZE 1048576

typedef struct {
    char* method;
    char* path;
    char* body;
    size_t content_length;
} HttpRequest;

typedef struct {
    int status_code;
    char* body;
    size_t content_length;
} HttpResponse;

static char* find_header_end(const char* buffer) {
    const char* end_marker = "\r\n\r\n";
    return strstr(buffer, end_marker);
}

static void parse_request(char* buffer, HttpRequest* req) {
    char* headers_end = find_header_end(buffer);
    if (!headers_end) return;
    
    char* headers = strdup(buffer);
    headers[headers_end - buffer] = '\0';
    
    char* first_line = strtok(headers, "\r\n");
    if (first_line) {
        char* space1 = strchr(first_line, ' ');
        if (space1) {
            *space1 = '\0';
            req->method = strdup(first_line);
            
            char* space2 = strchr(space1 + 1, ' ');
            if (space2) {
                *space2 = '\0';
                req->path = strdup(space1 + 1);
            }
        }
    }
    
    char* header;
    while ((header = strtok(NULL, "\r\n"))) {
        if (strncmp(header, "Content-Length: ", 16) == 0) {
            req->content_length = atoi(header + 16);
        }
    }
    
    req->body = strdup(headers_end + 4);
    if (!req->content_length) {
        req->content_length = strlen(req->body);
    }
    
    free(headers);
}

static void free_request(HttpRequest* req) {
    if (req) {
        free(req->method);
        free(req->path);
        free(req->body);
    }
}

static void write_response(socket_t client_fd, HttpResponse* resp) {
    char headers[BUFFER_SIZE];
    snprintf(headers, BUFFER_SIZE,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        resp->status_code,
        resp->status_code == 200 ? "OK" : "Bad Request",
        resp->content_length
    );
    
    send(client_fd, headers, (int)strlen(headers), 0);
    if (resp->body) {
        send(client_fd, resp->body, (int)resp->content_length, 0);
    }
}

static void handle_status(ServerConfig* config, HttpResponse* resp) {
    char* result = malloc(BUFFER_SIZE);
    snprintf(result, BUFFER_SIZE,
        "{"
        "\"status\":\"running\","
        "\"model\":{"
        "\"vocab_size\":%d,"
        "\"d_model\":%d,"
        "\"seq_len\":%d,"
        "\"cluster_size\":%d,"
        "\"num_layers\":%d,"
        "\"T_local\":%d"
        "},"
        "\"server\":{"
        "\"host\":\"%s\","
        "\"port\":%d"
        "}"
        "}",
        config->model->embedding->embed->shape[0],
        config->model->d_model,
        config->model->seq_len,
        config->model->layers[0]->cluster_size,
        config->model->num_layers,
        config->model->layers[0]->T_local,
        config->host,
        config->port
    );
    
    resp->status_code = 200;
    resp->body = result;
    resp->content_length = strlen(result);
}

static void handle_inference(ServerConfig* config, HttpRequest* req, HttpResponse* resp) {
    if (!req->body || req->content_length == 0) {
        resp->status_code = 400;
        resp->body = strdup("{\"error\":\"No input text provided\"}");
        resp->content_length = strlen(resp->body);
        return;
    }

    char input_text[BUFFER_SIZE];
    size_t text_len = req->content_length < BUFFER_SIZE - 1 ? req->content_length : BUFFER_SIZE - 1;
    strncpy(input_text, req->body, text_len);
    input_text[text_len] = '\0';

    int num_tokens;
    int* tokens = tokenize_text(config->tokenizer, input_text, config->model->seq_len, &num_tokens);
    if (!tokens) {
        resp->status_code = 500;
        resp->body = strdup("{\"error\":\"Failed to tokenize input\"}");
        resp->content_length = strlen(resp->body);
        return;
    }
    
    tensor_t* output = (tensor_t*)malloc(2 * sizeof(tensor_t));
    if (!output) {
        free(tokens);
        resp->status_code = 500;
        resp->body = strdup("{\"error\":\"Memory allocation failed\"}");
        resp->content_length = strlen(resp->body);
        return;
    }
    
    swarmformer_forward(config->model, tokens, output, 1, false);
    
    char* result = malloc(BUFFER_SIZE);
    if (!result) {
        free(tokens);
        free(output);
        resp->status_code = 500;
        resp->body = strdup("{\"error\":\"Memory allocation failed\"}");
        resp->content_length = strlen(resp->body);
        return;
    }
    
    snprintf(result, BUFFER_SIZE,
        "{\"input\":\"%s\",\"sentiment\":\"%s\",\"confidence\":%.3f}",
        input_text,
        output[1].f32 > output[0].f32 ? "positive" : "negative",
        output[1].f32 > output[0].f32 ? output[1].f32 : output[0].f32
    );
    
    resp->status_code = 200;
    resp->body = result;
    resp->content_length = strlen(result);
    
    free(tokens);
    free(output);
}

int start_server(ServerConfig* config) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        printf("Failed to initialize Winsock\n");
        return 1;
    }
#endif

    socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        perror("Socket creation failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }
    
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt)) == SOCKET_ERROR) {
        perror("Setsockopt failed");
#ifdef _WIN32
        CLOSE_SOCKET(server_fd);
        WSACleanup();
#endif
        return 1;
    }
    
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(config->host);
    address.sin_port = htons(config->port);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        perror("Bind failed");
#ifdef _WIN32
        CLOSE_SOCKET(server_fd);
        WSACleanup();
#endif
        return 1;
    }
    
    if (listen(server_fd, 10) == SOCKET_ERROR) {
        perror("Listen failed");
#ifdef _WIN32
        CLOSE_SOCKET(server_fd);
        WSACleanup();
#endif
        return 1;
    }
    
    printf("Server listening on %s:%d\n", config->host, config->port);
    printf("Available endpoints:\n");
    printf("  GET  /status    - Get server and model status\n");
    printf("  POST /inference - Run sentiment analysis\n");
    
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_fd == INVALID_SOCKET) {
            perror("Accept failed");
            continue;
        }
        
        char buffer[BUFFER_SIZE] = {0};
        int bytes_read = recv(client_fd, buffer, BUFFER_SIZE - 1, 0);
        
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            
            HttpRequest req = {0};
            HttpResponse resp = {0};
            
            parse_request(buffer, &req);
            
            if (req.method && req.path) {
                if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/status") == 0) {
                    handle_status(config, &resp);
                } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/inference") == 0) {
                    handle_inference(config, &req, &resp);
                } else {
                    resp.status_code = 400;
                    resp.body = strdup("{\"error\":\"Invalid request\"}");
                    resp.content_length = strlen(resp.body);
                }
            } else {
                resp.status_code = 400;
                resp.body = strdup("{\"error\":\"Malformed request\"}");
                resp.content_length = strlen(resp.body);
            }
            
            write_response(client_fd, &resp);
            
            free_request(&req);
            free(resp.body);
        }
        
        CLOSE_SOCKET(client_fd);
    }
    
#ifdef _WIN32
    CLOSE_SOCKET(server_fd);
    WSACleanup();
#endif
    return 0;
} 