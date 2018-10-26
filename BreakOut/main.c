//
//  main.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//

//Using SDL and standard IO
#include "SDL.h"
#include <stdio.h>

// Constants
#define GAME_WIDTH  160                    // Game dimension constants
#define GAME_HEIGHT 128

#define SCALE 4                                 // screen pixels per game pixel

#define SCREEN_WIDTH  (GAME_WIDTH*SCALE)     // Screen dimension constants
#define SCREEN_HEIGHT (GAME_HEIGHT*SCALE)

#define FRAME_DELAY 20                   // Frame delay (ms)
#define OUT_OF_PLAY 100                  // Ball outof play time (frames)

#define FACT 16                                 // ball position and velocity scale factor

// SDL objects
SDL_Window*   gWindow       = NULL;             // The window we'll be rendering to
SDL_Renderer* gRenderer     = NULL;             // The window renderer
SDL_Event e;                                    // Event handler


const Uint8* currentKeyStates;
int keystate = 0;

int x = (GAME_WIDTH/4)*FACT;                    // initial ball coordinates in fixed-point
int y = (GAME_HEIGHT/2)*FACT;

int u = 1*FACT;                                 // initial ball velocity in fixed-point
int v = -1*FACT;

int x_bat   = 0;                                // bat LHS x position
int bat_len = 8;                                // bat length in pixels

int frame_buff[GAME_WIDTH/8][GAME_HEIGHT];      // frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]

const int background = 0x1;                     // background colour - blue & soft
const int bat_col    = 0xE;                     // bat colour - yellow & hard
const int ball_col   = 0xF;                     // ball colour - white & hard
const int score_col  = 0x6;                     // score colour - yellow & soft

int out_of_play      = 0;                       // control pause when ball out of play
int score            = 0;                       // game score

const Uint16 digit[10] = {0x7B6F,0x1249,0x73E7,0x73CF,0x4979,0x79CF,0x79EF,0x7249,0x7BEF,0x7BCF};       // digits 0-9 in 3 x 5 format for score

void init_frame () {                                                                                    // initialise frame buffer to blue
    for (int i=0; i<(GAME_WIDTH/8); i++) {
        for (int j=0; j<GAME_HEIGHT; j++) {
            frame_buff[i][j] = 0x11111111 * background;
        }
    }
}

void set_pixel_col (int i, int j, int col) {                                                            // inserts pixel colour within word
    frame_buff[i/8][j] = (frame_buff[i/8][j] & ~(0xF << ((i%8)*4))) | (col << ((i%8)*4));
}

int get_pixel_col (int i, int j) {                                                                      // gets pixel colour from within word
    return (frame_buff[i/8][j] >> ((i%8)*4) & 0xF);
}


void update_frame (int keystate) {

// draw bat
    for (int i=0; i<bat_len; i++) set_pixel_col(i+x_bat, GAME_HEIGHT-1, background);                    // clear bat to background
    if (keystate & 1) if (--x_bat < 0) x_bat = 0;                                                       // move bat
    if (keystate & 2) if (++x_bat > GAME_WIDTH-bat_len) x_bat = GAME_WIDTH-bat_len;
    for (int i=x_bat; i<(x_bat+bat_len); i++) set_pixel_col(i, GAME_HEIGHT-1, bat_col);                 // draw yellow bat
    
// draw 3-digit score
    int s = score;
    for (int k=0; k<3; k++) {
        int s_x = GAME_WIDTH/2 + 3 - 4*k;
        for (int i=0; i<3; i++) {
            for (int j=0; j<5; j++) {
                if (digit[s%10] & (1<<(14-(i+3*j)))) set_pixel_col(s_x + i, 5 + j, score_col);
                else                                 set_pixel_col(s_x + i, 5 + j, background);
            }
        }
        s = s/10;
    }
    
// draw ball
    if (out_of_play == 0) {
        set_pixel_col(x/16, y/16, background);                                                          // clear pixel to background
        x += u;                                                                                         // move ball
        if ((x < -u) || (x >= ((GAME_WIDTH*16)-u)))  u = -u;                                            // bounce off sides
        y += v;
        if (y < -v) v = -v;                                                                             // bounce off top
    
//detect collision
        if (get_pixel_col(x/FACT, y/FACT) & 0x8) {                                                      // hit something hard!
            if (x/FACT < (x_bat+bat_len/4))        u = -FACT;
            else if (x/FACT < (x_bat+bat_len/2))   u = -FACT/2;
            else if (x/FACT < (x_bat+3*bat_len/4)) u = FACT/2;
            else                                   u = FACT;
            v = -FACT;
            y -= FACT;
            if (score < 999) ++score;
        }
    
// lost ball
        if (y >= (GAME_HEIGHT*FACT-v)) {
            v = -1*FACT;
            y = (GAME_HEIGHT/2)*FACT;
            out_of_play = OUT_OF_PLAY;
            if (score > 0) --score;
        } else {
            set_pixel_col(x/FACT, y/FACT, ball_col);                                                    // draw ball
        }
    } else --out_of_play;
}

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

int main( int argc, char* args[] )
{
    init_frame();
    int quit = init_SDL();
    currentKeyStates = SDL_GetKeyboardState (NULL);
    
    while (!quit) {
        
        update_frame(keystate);                                                                         // next game timestep

        for (int i=0; i<GAME_WIDTH; i++) {                                                              // display frame buffer
            for (int j=0; j<GAME_HEIGHT; j++) {
                SDL_Rect fillRect = { i*SCALE, j*SCALE, SCALE, SCALE };
                int col = get_pixel_col(i, j);
                SDL_SetRenderDrawColor( gRenderer, 0xFF * (col&4)/4, 0xFF * (col&2)/2, 0xFF * (col&1), 0xFF );
                SDL_RenderFillRect( gRenderer, &fillRect );
            }
        }
                    
        SDL_RenderPresent( gRenderer );                                                                 // update screen
        SDL_Delay(FRAME_DELAY);                                                                         // Wait frame delay time
        
        while (SDL_PollEvent(&e)) {                                                                     // collect inputs
            if (e.type == SDL_QUIT) quit = 1;
        }
        
        keystate = 0;                                                                                   // check L & R keys
        if (currentKeyStates[SDL_SCANCODE_LEFT])  keystate += 1;
        if (currentKeyStates[SDL_SCANCODE_RIGHT]) keystate += 2;
    }
    
    SDL_DestroyWindow( gWindow );                                                                       // Destroy window
    SDL_Quit();                                                                                         // Quit SDL subsystems
    return 0;
}