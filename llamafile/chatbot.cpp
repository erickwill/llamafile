// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <cosmo.h>
#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "llama.cpp/common.h"
#include "llama.cpp/llama.h"
#include "llamafile/bestline.h"
#include "llamafile/llamafile.h"

#define BOLD "\e[1m"
#define FAINT "\e[2m"
#define UNBOLD "\e[22m"
#define RED "\e[31m"
#define MAGENTA "\e[35m"
#define UNFOREGROUND "\e[39m"
#define BRIGHT_BLACK "\e[90m"
#define BRIGHT_RED "\e[91m"
#define BRIGHT_GREEN "\e[92m"
#define CLEAR_FORWARD "\e[K"

static int n_past;
static llama_model *g_model;
static llama_context *g_ctx;
static volatile sig_atomic_t g_got_sigint;

static void on_sigint(int sig) {
    g_got_sigint = 1;
}

static bool is_empty(const char *s) {
    int c;
    while ((c = *s++))
        if (!isspace(c))
            return false;
    return true;
}

static std::string basename(const std::string_view path) {
    size_t i, e;
    if ((e = path.size())) {
        while (e > 1 && path[e - 1] == '/')
            --e;
        i = e - 1;
        while (i && path[i - 1] != '/')
            --i;
        return std::string(path.substr(i, e - i));
    } else {
        return ".";
    }
}

static void on_completion(const char *line, bestlineCompletions *comp) {
    static const char *const kCompletions[] = {
        "/context", //
        "/stats", //
    };
    for (int i = 0; i < sizeof(kCompletions) / sizeof(*kCompletions); ++i)
        if (startswith(kCompletions[i], line))
            bestlineAddCompletion(comp, kCompletions[i]);
}

// handle irc style commands like: `/arg0 arg1 arg2`
static bool handle_command(const char *command) {
    if (!(command[0] == '/' && std::isalpha(command[1])))
        return false;
    std::vector<std::string> args;
    std::istringstream iss(command + 1);
    std::string arg;
    while (iss >> arg)
        args.push_back(arg);
    if (args[0] == "stats") {
        FLAG_log_disable = false;
        llama_print_timings(g_ctx);
        FLAG_log_disable = true;
    } else if (args[0] == "context") {
        int configured_context = llama_n_ctx(g_ctx);
        int max_context = llama_n_ctx_train(g_model);
        printf("%d out of %d context tokens used (%d tokens remaining)\n", n_past,
               configured_context, configured_context - n_past);
        if (configured_context < max_context)
            printf("use the `-c %d` flag at startup for maximum context\n", max_context);
    } else {
        printf("%s: unrecognized command\n", args[0].c_str());
    }
    return true;
}

static void print_logo(const char16_t *s) {
    for (int i = 0; s[i]; ++i) {
        switch (s[i]) {
        case u'█':
            printf(MAGENTA "█" UNFOREGROUND);
            break;
        case u'╚':
        case u'═':
        case u'╝':
        case u'╗':
        case u'║':
        case u'╔':
            printf(FAINT "%C" UNBOLD, s[i]);
            break;
        default:
            printf("%C", s[i]);
            break;
        }
    }
}

static void die_out_of_context(void) {
    fprintf(stderr,
            "\n" BRIGHT_RED
            "error: ran out of context window at %d tokens; you can use the maximum "
            "context window size by passing the flag `-c %d` to llamafile." UNFOREGROUND "\n",
            n_past, llama_n_ctx_train(g_model));
    exit(1);
}

static void eval_tokens(std::vector<llama_token> tokens, int n_batch) {
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
            n_eval = n_batch;
        if (llama_decode(g_ctx, llama_batch_get_one(&tokens[i], n_eval, n_past, 0)))
            die_out_of_context();
        n_past += n_eval;
    }
}

static void eval_id(int id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    eval_tokens(tokens, 1);
}

static void eval_string(const char *str, int n_batch, bool add_special, bool parse_special) {
    std::string str2 = str;
    eval_tokens(llama_tokenize(g_ctx, str2, add_special, parse_special), n_batch);
}

static void print_ephemeral(const char *description) {
    fprintf(stderr, " " BRIGHT_BLACK "%s" UNFOREGROUND "\r", description);
}

static void clear_ephemeral(void) {
    fprintf(stderr, CLEAR_FORWARD);
}

int chatbot_main(int argc, char **argv) {
    llamafile_check_cpu();
    ShowCrashReports();
    log_disable();

    gpt_params params;
    if (!gpt_params_parse(argc, argv, params))
        return 1;

    print_logo(u"\n\
██╗     ██╗      █████╗ ███╗   ███╗ █████╗ ███████╗██╗██╗     ███████╗\n\
██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║██║     ██╔════╝\n\
██║     ██║     ███████║██╔████╔██║███████║█████╗  ██║██║     █████╗\n\
██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══╝  ██║██║     ██╔══╝\n\
███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║     ██║███████╗███████╗\n\
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝\n");

    printf(BOLD "software" UNBOLD ": llamafile " LLAMAFILE_VERSION_STRING "\n" //
           BOLD "model" UNBOLD ":    %s\n\n",
           basename(params.model).c_str());

    print_ephemeral("initializing backend...");
    llama_backend_init();
    clear_ephemeral();

    print_ephemeral("initializing model...");
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = llamafile_gpu_layers(35);
    g_model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (g_model == NULL)
        return 2;
    clear_ephemeral();

    print_ephemeral("initializing context...");
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    if (g_ctx == NULL)
        return 3;
    clear_ephemeral();

    if (params.prompt.empty())
        params.prompt =
            "A chat between a curious human and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the human's questions.";

    print_ephemeral("loading prompt...");
    bool add_bos = llama_should_add_bos_token(llama_get_model(g_ctx));
    std::vector<llama_chat_msg> chat = {{"system", params.prompt}};
    std::string msg = llama_chat_apply_template(g_model, params.chat_template, chat, false);
    eval_string(msg.c_str(), params.n_batch, add_bos, true);
    clear_ephemeral();
    printf("%s\n", params.special ? msg.c_str() : params.prompt.c_str());

    // perform important setup
    struct llama_sampling_context *ctx_sampling = llama_sampling_init(params.sparams);
    signal(SIGINT, on_sigint);

    // run chatbot
    for (;;) {
        bestlineLlamaMode(true);
        bestlineSetCompletionCallback(on_completion);
        write(1, BRIGHT_GREEN, strlen(BRIGHT_GREEN));
        char *line = bestlineWithHistory(">>> ", "llamafile");
        write(1, UNFOREGROUND, strlen(UNFOREGROUND));
        if (!line)
            break;
        if (is_empty(line)) {
            free(line);
            continue;
        }
        if (handle_command(line)) {
            free(line);
            continue;
        }
        std::vector<llama_chat_msg> chat = {{"user", line}};
        std::string msg = llama_chat_apply_template(g_model, params.chat_template, chat, true);
        eval_string(msg.c_str(), params.n_batch, false, true);
        while (!g_got_sigint) {
            llama_token id = llama_sampling_sample(ctx_sampling, g_ctx, NULL);
            llama_sampling_accept(ctx_sampling, g_ctx, id, true);
            if (llama_token_is_eog(g_model, id))
                break;
            printf("%s", llama_token_to_piece(g_ctx, id, params.special).c_str());
            fflush(stdout);
            eval_id(id);
        }
        g_got_sigint = 0;
        printf("\n");
        free(line);
    }

    llama_sampling_free(ctx_sampling);
    llama_free(g_ctx);
    llama_free_model(g_model);
    llama_backend_free();
    return 0;
}
