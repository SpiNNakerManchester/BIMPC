//
//  bkout.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//
// Standard includes
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>

#include <recording.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
// **TODO** many of these magic numbers should be passed from Python
// Game dimension constants
#define GAME_WIDTH_MAX  160
#define GAME_HEIGHT_MAX 128

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//typedef enum
//{
//  REGION_SYSTEM,
//  REGION_BREAKOUT,
//  REGION_RECORDING,
//  REGION_PARAM,
//} region_t;

// Read param region
//address_t address = data_specification_get_data_address();
//address_t param_region = data_specification_get_region(REGION_PARAM, address);
//GAME_WIDTH_MAX = param_region[0]
//GAME_HEIGHT_MAX = param_region[1]

#define NUMBER_OF_LIVES 5
#define SCORE_DOWN_EVENTS_PER_DEATH 5

#define BRICKS_PER_ROW  5
#define BRICKS_PER_COLUMN  2


// Ball outof play time (frames)
#define OUT_OF_PLAY 100

// Frame delay (ms)
#define FRAME_DELAY 20 //14//20

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_BREAKOUT,
  REGION_RECORDING,
  REGION_PARAM,
} region_t;

typedef enum
{
  COLOUR_HARD       = 0x8,
  COLOUR_SOFT       = 0x0,
  COLOUR_BRICK      = 0x10,

  COLOUR_BACKGROUND = COLOUR_SOFT | 0x1,
  COLOUR_BAT        = COLOUR_HARD | 0x6,
  COLOUR_BALL       = COLOUR_HARD | 0x7,
  COLOUR_SCORE      = COLOUR_SOFT | 0x6,
  COLOUR_BRICK_ON   = COLOUR_BRICK | 0x0,
  COLOUR_BRICK_OFF  = COLOUR_BRICK | 0x1
} colour_t;

typedef enum
{
  KEY_LEFT  = 0x0,
  KEY_RIGHT = 0x1,
} key_t;

typedef enum
{
  SPECIAL_EVENT_SCORE_UP,
  SPECIAL_EVENT_SCORE_DOWN,
  SPECIAL_EVENT_MAX,
} special_event_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------


//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

static uint32_t _time;
uint32_t pkt_count;

int GAME_WIDTH = 160;
int GAME_HEIGHT = 128;
int game_bits = 8;

// initial ball coordinates in fixed-point
static int x; //= (GAME_WIDTH / 4) * FACT;
static int y; //= (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;

static int current_number_of_bricks;

static bool bricks[BRICKS_PER_COLUMN][BRICKS_PER_ROW];
bool print_bricks  = true;

int brick_corner_x=-1, brick_corner_y=-1;
int number_of_lives = NUMBER_OF_LIVES;

int x_factor = 1;
int y_factor = 1;

// ball position and velocity scale factor
int FACT = 16;

// initial ball velocity in fixed-point
int u = 16;// * FACT;
int v = -16;// * FACT;

// bat LHS x position
int x_bat = 40;

// bat length in pixels
int bat_len = 16;

// Brick parameters
int bricks_wide = 5;
int bricks_deep = 2;

int BRICK_WIDTH = 10;
int BRICK_HEIGHT = 6;

int BRICK_LAYER_OFFSET = 30;
int BRICK_LAYER_HEIGHT = 12;
int BRICK_LAYER_WIDTH = 160;

// frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]
static int frame_buff[GAME_WIDTH_MAX / 8][GAME_HEIGHT_MAX];

// control pause when ball out of play
static int out_of_play = 0;

// state of left/right keys
static int keystate = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;


//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//! How many ticks until next frame
static uint32_t tick_in_frame = 0;

uint32_t left_key_count = 0;
uint32_t right_key_count = 0;
uint32_t move_count_r = 0;
uint32_t move_count_l = 0;
uint32_t score_change_count=0;
int32_t current_score = 0;

//ratio used in randomising initial x coordinate
static uint32_t x_ratio=UINT32_MAX/(GAME_WIDTH_MAX);


//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void add_score_up_event()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_SCORE_UP), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Score up\n");
  current_score++;
}

static inline void add_score_down_event()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_SCORE_DOWN), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Score down\n");
  current_score--;
}

void add_event(int i, int j, colour_t col, bool bricked)
{
    const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
    const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << (game_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit);

//    log_debug("e %d, %d, %u, %08x, b%d", i, j, col, spike_key, game_bits);
    spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
}

// gets pixel colour from within word
static inline colour_t get_pixel_col (int i, int j)
{
  return (colour_t)(frame_buff[i / 8][j] >> ((i % 8)*4) & 0xF);
}

// inserts pixel colour within word
static inline void set_pixel_col (int i, int j, colour_t col, bool bricked)
{
    io_printf(IO_BUF, "setting (%d,%d) to %d, b-%d, g%d, w%d, h%d\n", i, j, col, bricked, game_bits, GAME_WIDTH, GAME_HEIGHT);
    if (bricked) {
        add_event((brick_corner_x * BRICK_WIDTH),
                      (brick_corner_y* BRICK_HEIGHT + BRICK_LAYER_OFFSET),
                      COLOUR_BACKGROUND, bricked);
    }
    else if (col != get_pixel_col(i, j))
    {
      /*  //just update bat pixels in game frame
        if (j==GAME_HEIGHT-1)
        {
            frame_buff[i / 8][j] = (frame_buff[i / 8][j] & ~(0xF << ((i % 8) * 4))) | ((int)col << ((i % 8)*4));
        }
        else
        {
            frame_buff[i / 8][j] = (frame_buff[i / 8][j] & ~(0xF << ((i % 8) * 4))) | ((int)col << ((i % 8)*4));
            add_event (i, j, col);
        }*/
        frame_buff[i / 8][j] = (frame_buff[i / 8][j] & ~(0xF << ((i % 8) * 4))) | ((int)col << ((i % 8)*4));
        add_event (i, j, col, bricked);
    }
}

static inline bool is_a_brick(int x, int y) // x - width, y- height?
{
    int pos_x=0, pos_y=0;

    if ( y >= BRICK_LAYER_OFFSET && y < BRICK_LAYER_OFFSET + BRICK_LAYER_HEIGHT) {
        pos_x = x / BRICK_WIDTH;
        pos_y = (y - BRICK_LAYER_OFFSET) / BRICK_HEIGHT;
        bool val = bricks[pos_y][pos_x];
//        if (pos_y>= BRICKS_PER_COLUMN) {
//            log_error("%d", pos_y);
//            rt_error(RTE_SWERR);
//        }
//        if (pos_x>= BRICKS_PER_ROW) {
//            log_error("%d", pos_x);
//            rt_error(RTE_SWERR);
//        }
        bricks[pos_y][pos_x] = false;
        if (val) {
            brick_corner_x = pos_x;
            brick_corner_y = pos_y;
            current_number_of_bricks--;
        }
        else {
            brick_corner_x = -1;
            brick_corner_y = -1;
        }


//        io_printf(IO_BUF, "%d %d %d %d\n", x, y, pos_x, pos_y);
        return val;
    }
    brick_corner_x = -1;
    brick_corner_y = -1;
    return false;
}



//----------------------------------------------------------------------------
// Static functions
//----------------------------------------------------------------------------
// initialise frame buffer to blue
static void init_frame ()
{
    for (int i=0; i<(GAME_WIDTH/8); i++)
    {
        for (int j=0; j<GAME_HEIGHT; j++)
        {
            frame_buff[i][j] = 0x11111111 * COLOUR_BACKGROUND;
        }
    }

    for (int i =0; i<BRICKS_PER_COLUMN; i++)
        for (int j=0; j<BRICKS_PER_ROW; j++) {
            bricks[i][j] = true;
        }
    current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
}

static void update_frame ()
{
    // draw bat
    // Cache old bat position
    const uint32_t old_xbat = x_bat;
    int move_direction;
    if (right_key_count > left_key_count){
        move_direction = KEY_RIGHT;
        move_count_r++;
    //    io_printf(IO_BUF, "moved right\n");
    }
    else if (left_key_count > right_key_count){
        move_direction = KEY_LEFT;
        move_count_l++;
        //    io_printf(IO_BUF, "moved left\n");
    }
    else{
        move_direction = 2;
        //    io_printf(IO_BUF, "didn't move!\n");
    }
    io_printf(IO_BUF, "left = %d, right = %d\n", left_key_count, right_key_count);


    // Update bat and clamp
    if (move_direction == KEY_LEFT && --x_bat < 0)
    {
        x_bat = 0;
    }
    else if (move_direction == KEY_RIGHT && ++x_bat > GAME_WIDTH-bat_len-1)
    {
        x_bat = GAME_WIDTH-bat_len-1;
    }

    // Clear keystate
    left_key_count = 0;
    right_key_count = 0;

    // If bat's moved
    if (old_xbat != x_bat)
    {
        // Draw bat pixels
        for (int i = x_bat; i < (x_bat + bat_len); i++)
        {
            set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT, false);
        }



        // Remove pixels left over from old bat
        if (x_bat > old_xbat)
        {
            set_pixel_col(old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND, false);
        }
        else if (x_bat < old_xbat)
        {
            set_pixel_col(old_xbat + bat_len, GAME_HEIGHT-1, COLOUR_BACKGROUND, false);
        }

        //only draw left edge of bat pixel
        // add_event(x_bat, GAME_HEIGHT-1, COLOUR_BAT);
        //send off pixel to network (ignoring game frame buffer update)
        // add_event (old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND);
    }

    // draw ball
    if (out_of_play == 0)
    {
        // clear pixel to background
        io_printf(IO_BUF, "setting ball to background x=%d, y=%d, fact=%d, xf=%d, yf=%d\n", x, y, FACT, x/FACT, y/FACT);
        set_pixel_col(x/FACT, y/FACT, COLOUR_BACKGROUND, false);

        // move ball in x and bounce off sides
        x += u;
        if (x < -u)
        {
            //      io_printf(IO_BUF, "OUT 1\n");
            u = -u;
        }
        if (x >= ((GAME_WIDTH*FACT)-u))
        {
            //      io_printf(IO_BUF, "OUT 2 x = %d, u = %d, gw = %d, fact = %d\n", x, u, GAME_WIDTH, FACT);
            u = -u;
        }

        // move ball in y and bounce off top
        y += v;
        // if ball entering bottom row, keep it out XXX SD
        if (y == GAME_HEIGHT-1)
        {
            y = GAME_HEIGHT;
        }
        if (y < -v)
        {
            v = -v;
        }

        io_printf(IO_BUF, "about to is a brick\n");
        //detect collision
        // if we hit something hard! -- paddle or brick
        bool bricked = is_a_brick(x/ FACT, y/ FACT);

        if ( bricked ) {
            io_printf(IO_BUF, "got in bricked\n");
            int brick_x = brick_corner_x * BRICK_WIDTH;
            int brick_y = (brick_corner_y* BRICK_HEIGHT + BRICK_LAYER_OFFSET);
            //        io_printf(IO_BUF, "x-brick_x = %d, %d %d\n",x/FACT - brick_x, x/FACT, brick_x);
            //        io_printf(IO_BUF, "y-brick_y = %d, %d %d",y/FACT - brick_y, y/FACT, brick_y);

            if ( brick_x == x/FACT && u > 0){
                u = -u;
            }
            else if (x/FACT == brick_x + BRICK_WIDTH - 1 && u < 0){
                u = -u;
            }
            if (brick_y  == y/FACT && v > 0){
                v = -v;
            }
            if (y/FACT ==  brick_y + BRICK_HEIGHT - 1 && v < 0){
                v = -v;
            }

            set_pixel_col(x/FACT, y/FACT, COLOUR_BACKGROUND, bricked);

            bricked= false;
            // Increase score
            add_score_up_event();
        }


        if (get_pixel_col(x / FACT, y / FACT) & COLOUR_HARD && y > GAME_HEIGHT*(FACT / 2))
        {
            io_printf(IO_BUF, "got in get pixel colour\n");
            bool broke = false;
            if (x/FACT < (x_bat+bat_len/4))
            {
                //        log_info("BAT 1");
                u = -FACT;
            }
            else if (x/FACT < (x_bat+bat_len/2))
            {
                //        log_info("BAT 2");
                u = -FACT/2;
            }
            else if (x/FACT < (x_bat+3*bat_len/4))
            {
                //        log_info("BAT 3");
                u = FACT/2;
            }
            else if (x/FACT < (x_bat+bat_len))
            {
                //        log_info("BAT 4");
                u = FACT;
            }
            else
            {
                io_printf(IO_BUF, "Broke bat 0x%x\n", (frame_buff[(x/FACT) / 8][y/FACT] >> ((x/FACT % 8)*4) & 0xF));
                broke = true;
                //        u = FACT;
            }

            //     if (bricked) {
            //        set_pixel_col(x/FACT, y/FACT, COLOUR_BACKGROUND, bricked);
            //     }
            if (broke == false)
            {
              v = -FACT;
              y -= FACT;
            }
            // Increase score
            //      add_score_up_event();
        }

        // lost ball
        if (y >= (GAME_HEIGHT*FACT-v))
        {
            io_printf(IO_BUF, "got in lost ball\n");
            v = -1 * FACT;
            y = (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;

            if(mars_kiss32() > 0xFFFF){
                //        log_info("MARS 1");
                u = -u;
            }

            //randomises initial x location
            x = GAME_WIDTH;

            while (x >= GAME_WIDTH)
                x = (int)(mars_kiss32()/x_ratio);
            //      x = (int)(mars_kiss32()%GAME_WIDTH);
            //      log_info("random x = %d", x);
            x *= FACT;

            out_of_play = OUT_OF_PLAY;
            // Decrease score
            number_of_lives--;
            if (!number_of_lives){
                for(int i=0; i<SCORE_DOWN_EVENTS_PER_DEATH;i++) {
                    add_score_down_event();
                }
                number_of_lives = NUMBER_OF_LIVES;
            }
            else {
                add_score_down_event();
            }
        }
        // draw ball
        else
        {
            io_printf(IO_BUF, "else\n");
            set_pixel_col(x/FACT, y/FACT, COLOUR_BALL, false);
        }
    }
    else
    {
        --out_of_play;
    }
}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise breakout: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address))
    {
        return false;
    }
    /*
    simulation_initialise(
        address_t address, uint32_t expected_app_magic_number,
        uint32_t* timer_period, uint32_t *simulation_ticks_pointer,
        uint32_t *infinite_run_pointer, int sdp_packet_callback_priority,
        int dma_transfer_done_callback_priority)
    */
    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL))
    {
        return false;
    }
    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read breakout region
    address_t breakout_region = data_specification_get_region(REGION_BREAKOUT, address);
    key = breakout_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    //get recording region
    address_t recording_address = data_specification_get_region(
                                       REGION_RECORDING,address);

    // Read param region
    address_t param_region = data_specification_get_region(REGION_PARAM, address);

    x_factor = param_region[0];
    y_factor = param_region[1];

//    int *GAME_WIDTH_POINTER;
//    GAME_WIDTH_POINTER = &GAME_WIDTH;
//    *GAME_WIDTH_POINTER = param_region[0];
    GAME_WIDTH = GAME_WIDTH / x_factor;
//    int *GAME_HEIGHT_POINTER;
//    GAME_HEIGHT_POINTER = &GAME_HEIGHT;
//    *GAME_HEIGHT_POINTER = param_region[1];
    GAME_HEIGHT = GAME_HEIGHT / y_factor;

    x = (GAME_WIDTH / 4) * FACT;
    y = (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;
//    frame_buff[GAME_WIDTH / 8][GAME_HEIGHT];
    x_ratio=UINT32_MAX/(GAME_WIDTH);

    // rescale variables
    FACT = FACT / y_factor;

    u = 1 * FACT;
    v = -1 * FACT;

    x_bat = x_bat / x_factor;

    bat_len = bat_len / x_factor;

    BRICK_WIDTH = GAME_WIDTH / bricks_wide;//BRICK_WIDTH / x_factor;
    BRICK_HEIGHT = 16 / y_factor;//BRICK_HEIGHT / y_factor;

    BRICK_LAYER_OFFSET = BRICK_LAYER_OFFSET / y_factor;
    BRICK_LAYER_HEIGHT = bricks_deep * BRICK_HEIGHT;//BRICK_LAYER_HEIGHT / y_factor;
    BRICK_LAYER_WIDTH = BRICK_WIDTH;//BRICK_LAYER_WIDTH / x_factor;

    io_printf(IO_BUF, "bw%d, bh%d, blo%d, blh%d, blw%d, xb%d, bl%d, u%d, v%d\n", BRICK_WIDTH, BRICK_HEIGHT, BRICK_LAYER_OFFSET, BRICK_LAYER_HEIGHT, BRICK_LAYER_WIDTH, x_bat, bat_len, u, v);

    int *game_bits_pointer;
    game_bits_pointer = &game_bits;
    *game_bits_pointer = ceil(log2(GAME_WIDTH));
    game_bits = ceil(log2(GAME_WIDTH));

    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(recording_address, &recording_flags))
    {
        rt_error(RTE_SWERR);
        return false;
    }

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

//----------------------------------------------------------------------------
// Callbacks
//----------------------------------------------------------------------------
// incoming SDP message
/*void process_sdp (uint m, uint port)
*{
    sdp_msg_t *msg = (sdp_msg_t *) m;

    io_printf (IO_BUF, "SDP len %d, port %d - %s\n", msg->length, port, msg->data);
    // Port 1 - key data
    if (port == 1) spin1_memcpy(&keystate, msg->data, 4);
    spin1_msg_free (msg);
    if (port == 7) spin1_exit (0);
}*/

void resume_callback() {
    recording_reset();
}

void timer_callback(uint unused, uint dummy)
{
  use(unused);
  use(dummy);
  // If a fixed number of simulation ticks are specified and these have passed
  //
//  ticks++;
    //this makes it count twice, WTF!?

  _time++;
  score_change_count++;

//   if (!current_number_of_bricks) {
//        for (int i =0; i<BRICKS_PER_COLUMN; i++)
//            for (int j=0; j<BRICKS_PER_ROW; j++) {
//                bricks[i][j] = true;
//                }
//          current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
//          print_bricks = true;
//          v = -1 * FACT;
//      y = (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;
//
//      if(mars_kiss32() > 0xFFFF){
//        u = -u;
//      }
//
//      //randomises initial x location
//      x = GAME_WIDTH;
//
//      while (x >= GAME_WIDTH)
//         x = (int)(mars_kiss32()/x_ratio);
////      x = (int)(mars_kiss32()%GAME_WIDTH);
////      io_printf(IO_BUF, "random x = %d", x);
//      x *= FACT;
//   }
//
//   if (print_bricks) {
//    print_bricks = false;
//    for (int i =0; i<BRICKS_PER_COLUMN; i++)
//        for (int j=0; j<BRICKS_PER_ROW; j++) {
//            if (bricks[i][j]) {
//                add_event(j * BRICK_WIDTH,
//                              i* BRICK_HEIGHT + BRICK_LAYER_OFFSET,
//                              COLOUR_BRICK_ON, true);
//
//            }
//        }
//    io_printf(IO_BUF, "printed bricks");
//   }

  if (!infinite_run && _time >= simulation_ticks)
  {
    //spin1_pause();
    recording_finalise();
    // go into pause and resume state to avoid another tick
    simulation_handle_pause_resume(resume_callback);
//    spin1_callback_off(MC_PACKET_RECEIVED);

    io_printf(IO_BUF, "move count Left %u\n", move_count_l);
    io_printf(IO_BUF, "move count Right %u\n", move_count_r);
    io_printf(IO_BUF, "infinite_run %d; time %d\n",infinite_run, _time);
    io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);
//    io_printf(IO_BUF, "key count Left %u", left_key_count);
//    io_printf(IO_BUF, "key count Right %u", right_key_count);

    io_printf(IO_BUF, "Exiting on timer.\n");
    simulation_ready_to_read();

    _time -= 1;
    return;
  }
  // Otherwise
  else
  {
    // Increment ticks in frame counter and if this has reached frame delay
    tick_in_frame++;
    if(tick_in_frame == FRAME_DELAY)
    {
      if (!current_number_of_bricks) {
        for (int i =0; i<BRICKS_PER_COLUMN; i++)
            for (int j=0; j<BRICKS_PER_ROW; j++) {
                bricks[i][j] = true;
                }
          current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
//          print_bricks = true;
          v = -1 * FACT;
      y = (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;

      if(mars_kiss32() > 0xFFFF){
//        log_info("MARS 2");
        u = -u;
      }

      //randomises initial x location
      x = GAME_WIDTH;

      while (x >= GAME_WIDTH)
         x = (int)(mars_kiss32()/x_ratio);
//      x = (int)(mars_kiss32()%GAME_WIDTH);
//      log_info("random x = %d", x);
      x *= FACT;
      }

//       if (print_bricks) {
//        print_bricks = false;
        for (int i =0; i<BRICKS_PER_COLUMN; i++)
            for (int j=0; j<BRICKS_PER_ROW; j++) {
                if (bricks[i][j]) {
                    add_event(j * BRICK_WIDTH,
                                  i* BRICK_HEIGHT + BRICK_LAYER_OFFSET,
                                  COLOUR_BRICK_ON, true);

                }
            }
//        log_info("printed bricks");
//       }
      //log_info("pkts: %u   L: %u   R: %u", pkt_count, move_count_l, move_count_r);
      // If this is the first update, draw bat as
      // collision detection relies on this
      if(_time == FRAME_DELAY)
      {
        // Draw bat
        for (int i = x_bat; i < (x_bat + bat_len); i++)
        {
          set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT, false);
        }
      }

      // Reset ticks in frame and update frame
      tick_in_frame = 0;
      //print_bricks = true;
      update_frame();
      // Update recorded score every 10s
      if(score_change_count>=10000){
        recording_record(0, &current_score, 4);
        score_change_count=0;
      }
    }
  }
//  log_info("time %u", ticks);
//  log_info("time %u", _time);
}

void mc_packet_received_callback(uint key, uint payload)
{
  use(payload);
//  log_info("mc pack in %u", key);

 /* uint stripped_key = key & 0xFFFFF;
  pkt_count++;

  // Left
  if(stripped_key & KEY_LEFT)
  {
    left_key_count++;
  }
  // Right
  else if (stripped_key & KEY_RIGHT)
  {
    right_key_count++;
  }*/
/*
  if(key & KEY_RIGHT){
    right_key_count++;
  }
  // Left
//  if(key & KEY_LEFT){
//  else{
  else if(key & KEY_LEFT){
    left_key_count++;
  }
//  else
/*/
  // Right
  if(key & KEY_RIGHT){
    right_key_count++;
  }
  else {
//  else{
    left_key_count++;
  }
//*/
//  log_info("mc key %u, L %u, R %u", key, left_key_count, right_key_count);
}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


void rte_handler (uint code)
{
  // Save code

  sark.vcpu->user0 = code;
  sark.vcpu->user1 = (uint) sark.sdram_buf;

  // Copy ITCM to SDRAM

  sark_word_cpy (sark.sdram_buf, (void *) ITCM_BASE, ITCM_SIZE);

  // Copy DTCM to SDRAM

  sark_word_cpy (sark.sdram_buf + ITCM_SIZE, (void *) DTCM_BASE, DTCM_SIZE);

  // Try to re-establish consistent SARK state

  sark_vic_init ();

  sark_vic_set ((vic_slot) sark_vec->sark_slot, CPU_INT, 1, sark_int_han);

  uint *stack = sark_vec->stack_top - sark_vec->svc_stack;

  stack = cpu_init_mode (stack, IMASK_ALL+MODE_IRQ, sark_vec->irq_stack);
  stack = cpu_init_mode (stack, IMASK_ALL+MODE_FIQ, sark_vec->fiq_stack);
  (void)  cpu_init_mode (stack, IMASK_ALL+MODE_SYS, 0);

  cpu_set_cpsr (MODE_SYS);

  // ... and sleep

  while (1)
    cpu_wfi ();
}

//-------------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
    // Load DTCM data
    uint32_t timer_period;
    if (!initialize(&timer_period))
    {
        io_printf(IO_BUF,"Error in initialisation - exiting!\n");
        rt_error(RTE_SWERR);
        return;
    }


    // frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]
//    static int frame_buff[GAME_WIDTH / 8][GAME_HEIGHT];

    init_frame();
    keystate = 0; // IDLE
    tick_in_frame = 0;
    pkt_count = 0;

    // Set timer tick (in microseconds)
    io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
    spin1_set_timer_tick(timer_period);
    io_printf(IO_BUF, "bricks %x\n", &bricks);

    io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, 2);
    spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

    _time = UINT32_MAX;

    simulation_run();




}
