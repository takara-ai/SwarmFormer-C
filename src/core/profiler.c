void print_progress_bar(int current, int total, int width) {
    float progress = (float)current / total;
    int filled = (int)(progress * width);
    
    printf("\rProgress: [");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else printf(" ");
    }
    printf("] %3d%%", (int)(progress * 100));
    fflush(stdout);
    
    if (current == total) printf("\n");
}

void start_profile(const char* name) {
    if (!profiler.enabled) return;
    
    ProfileEntry* entry = &profiler.entries[profiler.num_entries++];
    strncpy(entry->name, name, MAX_NAME_LENGTH - 1);
    entry->name[MAX_NAME_LENGTH - 1] = '\0';
    entry->start_time = get_time();
    entry->flops = 0;
}

void end_profile(const char* name, double flops) {
    if (!profiler.enabled) return;
    
    double end_time = get_time();
    
    for (int i = profiler.num_entries - 1; i >= 0; i--) {
        if (strcmp(profiler.entries[i].name, name) == 0) {
            profiler.entries[i].duration = end_time - profiler.entries[i].start_time;
            profiler.entries[i].flops = flops;
            break;
        }
    }
    
    if (profiler.current_iter > 0) {
        print_progress_bar(profiler.current_iter, profiler.total_iters, 50);
    }
}

void reset_profiler() {
    profiler.num_entries = 0;
    profiler.enabled = false;
    profiler.current_iter = 0;
    profiler.total_iters = 0;
}

void start_benchmark(int num_iterations) {
    reset_profiler();
    profiler.enabled = true;
    profiler.total_iters = num_iterations;
    printf("Starting benchmark with %d iterations...\n", num_iterations);
}

void end_benchmark() {
    if (!profiler.enabled) return;
    
    printf("\nBenchmark Results:\n");
    printf("%-30s %15s %15s %15s\n", "Operation", "Time (ms)", "GFLOPS", "Iterations");
    printf("%-30s %15s %15s %15s\n", "---------", "---------", "------", "----------");
    
    for (int i = 0; i < profiler.num_entries; i++) {
        ProfileEntry* entry = &profiler.entries[i];
        if (entry->duration > 0) {
            double avg_time_ms = (entry->duration * 1000.0) / profiler.total_iters;
            double gflops = (entry->flops * 1e-9) / entry->duration;
            printf("%-30s %15.3f %15.2f %15d\n", 
                   entry->name, avg_time_ms, gflops, profiler.total_iters);
        }
    }
    
    reset_profiler();
} 