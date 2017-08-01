//
//  main.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//

//Using SDL, standard IO, and other stuff
#include "SDL.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

SDL_Window*   gWindow       = NULL;             // The window we'll be rendering to
SDL_Renderer* gRenderer     = NULL;             // The window renderer
SDL_Event e;                                    // Event handler

const Uint8* currentKeyStates;

#define GAME_WIDTH  160                    // Game dimension constants
#define GAME_HEIGHT 128

#define SCALE 4                                 // screen pixels per game pixel

#define SCREEN_WIDTH  (GAME_WIDTH*SCALE)     // Screen dimension constants
#define  SCREEN_HEIGHT (GAME_HEIGHT*SCALE)

int events[60];                                // display update events; events[0] = number of events to process

const int background = 0x1;                     // background colour - blue & soft

int init_SDL () {
    int quit = 0;                                                                                       // Main loop flag
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {                                                                 // Initialize SDL
        printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
        quit = 1;
    } else {                                                                                            // Create window
        gWindow = SDL_CreateWindow( "BreakOut!", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );
        if ( gWindow == NULL ) {
            printf( "Window could not be created! SDL_Error: %s\n", SDL_GetError() );
            quit = 1;
        } else {
            gRenderer = SDL_CreateRenderer( gWindow, -1, SDL_RENDERER_ACCELERATED );                    // Create renderer for window
            if( gRenderer == NULL ) {
                printf( "Renderer could not be created! SDL Error: %s\n", SDL_GetError() );
                quit = 1;
            }
        }
    }
    return quit;
}

int disp_buff[GAME_WIDTH/8][GAME_HEIGHT];                                                               // display buffer - should be a copy of frame buffer!

int get_disp_pixel_col (int i, int j) {                                                                 // gets display pixel colour from within word
    return (disp_buff[i/8][j] >> ((i%8)*4) & 0xF);
}

void init_disp () {                                                                                     // initialise display buffer to blue
    for (int i=0; i<(GAME_WIDTH/8); i++) {
        for (int j=0; j<GAME_HEIGHT; j++) {
            disp_buff[i][j] = 0x11111111 * background;
        }
    }
}

#define SpiNN3 "192.168.240.253"
#define HOST   "192.168.240.254"

#define MAXBUFLEN 1024
char buf[MAXBUFLEN];

struct addrinfo hints, *resTx, *resRx;

void *get_in_addr(struct sockaddr *sa) {                                                                // get sockaddr, IPv4 or IPv6:
    if (sa->sa_family == AF_INET) return &(((struct sockaddr_in*)sa)->sin_addr);
    else                          return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int make_sock (uint16_t port, struct addrinfo *res) {
    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);                            // make a socket:
    if (sockfd < 0) printf ("Socket failed to set up...\n");
    struct sockaddr_in local_address;
    local_address.sin_family = AF_INET;
    local_address.sin_addr.s_addr = htonl(INADDR_ANY);
    local_address.sin_port = htons(port);
    if (bind(sockfd, (struct sockaddr *) &local_address, sizeof(local_address)) < 0) {
                    printf("Error binding to local address\n");
    }
    return sockfd;
}

int init_SDP_Tx () {
    if (getaddrinfo(SpiNN3, "17893", &hints, &resTx) != 0)      printf ("Getaddr failed...\n");
    int sockfd = make_sock (17893, resTx);
    if (connect(sockfd, resTx->ai_addr, resTx->ai_addrlen) < 0) printf ("Socket %s failed to connect...\n", "17893");
    else                                                        printf ("Socket %s connected...\n", "17893");
    return sockfd;
}

int init_SDP_Rx () {
    if (getaddrinfo(SpiNN3, "17894", &hints, &resRx) != 0)      printf ("Getaddr failed...\n");
    int sockfd = make_sock (17894, resRx);
    return sockfd;
}

void send_keystate (int sockfdTx, int keystate) {                                                       // SDP out - just send 4 bytes of keystate - sendto()
    char msg[31] = {0x00, 0x00, 0x07, 0x01, 0x21, 0x20, 0x00, 0x00, 0x00, 0x00,                         // IPtag 1, to CPU 1, port 1, to chip (0,0); from don't care!
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,                                     // 16 bytes of SCP header
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    msg[26] =  keystate        & 0xff;
    msg[27] = (keystate >>  8) & 0xff;
    msg[28] = (keystate >> 16) & 0xff;
    msg[29] = (keystate >> 24)        & 0xff;
    msg[30] =  0x00;
    printf("waiting to send...\n");
    ssize_t bytes_sent = send(sockfdTx, &msg, 31, 0);
    printf("packet sent...\n");
    if (bytes_sent < 0) printf ("packet send failed...\n");
}

void receive_events (int sockfdRx) {                                                                    // SDP in - receive events[0..events[0]] - read()
    printf("waiting to receive...\n");
    ssize_t bytes_received = recv(sockfdRx, &buf, MAXBUFLEN-1, 0);
    printf("packet received...\n");
    if (bytes_received < 0) printf ("junk packet received...\n");                                       // panic!!
    events[0] = buf[26] + (buf[27]<<8) + (buf[28]<<16) + (buf[29]<<24);                                 // skip 26 bytes for SDP & SCP headers + UDP alignment padding
    for (int i=1; i<=events[0]; i++) events[i] = buf[26+4*i] + (buf[27+4*i]<<8) + (buf[28+4*i]<<16) + (buf[29+4*i]<<24);
}

void process_events () {
    if (events[0]==0) return;
    for (int k=1; k<=events[0]; k++ ) {
        int i   = (events[k] >> 16) & 0xff;
        int j   = (events[k] >>  8) & 0xff;
        int col =  events[k]        & 0xf;
        disp_buff[i/8][j] = (disp_buff[i/8][j] & ~(0xF << ((i%8)*4))) | (col << ((i%8)*4));
    }
}

int main( int argc, char* args[] )
{
    printf("Starting...\n");

    memset(&hints, 0, sizeof hints);                                                                    // load up address structs with getaddrinfo():
    hints.ai_family   = AF_INET;                                                                        // use IPv4
    hints.ai_socktype = SOCK_DGRAM;                                                                     // DGRAM or STREAM
    hints.ai_flags    = AI_PASSIVE;                                                                     // fill in my IP for me

    int sockfdTx = init_SDP_Tx();
    int sockfdRx = init_SDP_Rx();
    init_disp();
    int quit = init_SDL();
    currentKeyStates = SDL_GetKeyboardState (NULL);
    
    while (!quit) {

        receive_events(sockfdRx);
        process_events();                                                                               // use events to update display buffer

        for (int i=0; i<GAME_WIDTH; i++) {                                                              // display frame buffer
            for (int j=0; j<GAME_HEIGHT; j++) {
                SDL_Rect fillRect = { i*SCALE, j*SCALE, SCALE, SCALE };
                int col = get_disp_pixel_col(i, j);
                SDL_SetRenderDrawColor( gRenderer, 0xFF * (col&4)/4, 0xFF * (col&2)/2, 0xFF * (col&1), 0xFF );
                SDL_RenderFillRect( gRenderer, &fillRect );
            }
        }
                    
        SDL_RenderPresent( gRenderer );                                                                 // update screen
        
        while (SDL_PollEvent(&e)) {                                                                     // collect inputs
            if (e.type == SDL_QUIT) quit = 1;
        }
        
        int keystate = 0;                                                                               // check L & R keys
        if (currentKeyStates[SDL_SCANCODE_LEFT])  keystate += 1;
        if (currentKeyStates[SDL_SCANCODE_RIGHT]) keystate += 2;
        send_keystate (sockfdTx, keystate);
    }
    
    SDL_DestroyWindow( gWindow );                                                                       // Destroy window
    SDL_Quit();                                                                                         // Quit SDL subsystems
    close(sockfdTx);                                                                                    // close sockets
    close(sockfdRx);
    return 0;
}
