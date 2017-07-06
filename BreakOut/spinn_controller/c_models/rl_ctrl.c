//
//  Re-inforcement learning for BreakOut
//
//
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

typedef enum
{
  REGION_SYSTEM,
  REGION_BREAKOUT,
  REGION_PROVENANCE,
} region_t;

typedef enum
{
  SPECIAL_EVENT_SCORE_UP,
  SPECIAL_EVENT_SCORE_DOWN,
  SPECIAL_EVENT_MAX,
} special_event_t;

typedef enum
{
  KEY_LEFT  = 0x1,
  KEY_RIGHT = 0x2,
  KEY_NONE  = 0x3,
} key_t;

#define GAME_WIDTH  160
#define GAME_HEIGHT 128

#define RLSTATE_WIDTH  (GAME_WIDTH>>4)
#define RLSTATE_HEIGHT (GAME_HEIGHT>>4)
#define NUM_ACTIONS 4

#define numGrids 1

#define TICKS_PER_RESPONSE 100

#define STATEVECTOR_NUM_WORDS 1

#define CONSECUTIVE_MOVES 10 


// **** Change scale shift to change the coarseness of the state space ****
// Image (and hence each state dimension) is scaled by 2 ** scale_shift.
int scale_shift = 5;
int scale;
accum discount_rate = 0.8;
accum alpha = 0.01;
accum lambda = 0.95;
static uint32_t key;
uint32_t counter;
uint32_t ticks;
uint32_t score_up, score_down; 
uint32_t cum_score_up, cum_score_down;
accum    recent_reward_or_punishment;
uint32_t new_key;
uint32_t color_bit;
uint32_t sz;
int32_t change_y, change_x;
int32_t ball_y, ball_x;
int32_t bat_x;
int32_t prev_ball_y, prev_ball_x;
int32_t prev_bat_x;
uint8_t  * pScreen[numGrids];
accum *  pVbase;
accum *  pQbase;
accum *  pEligbase;
accum    v_value, q_value;
accum    q_increment;
accum    prev_v_value;
uint32_t prev_state_index;
uint32_t state_index;
accum    prev_q_value   = 0.0;
uint32_t q_index        = 0;
uint32_t prev_q_index   = 0;
uint32_t prev_action    = KEY_NONE;
uint32_t current_action = KEY_NONE;
uint32_t next_action    = KEY_NONE;
uint32_t move_count;
uint32_t outcount       = 0;
uint32_t value_statespace_elements;
uint32_t action_elements, elements_per_action;
accum    q_left, q_right, q_none;
accum    q_left_n, q_right_n, q_none_n;
uint32_t move_direction;
uint32_t total_move_count = 0;
uint32_t left_total = 0;
uint32_t right_total = 0;
uint32_t none_total = 0;
uint32_t bits_for_width;
uint32_t bits_for_height;
uint32_t bat_x_bit_start;
uint32_t ball_x_bit_start;
uint32_t ball_y_bit_start;
uint32_t action_bit_start;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

//! the number of timer ticks that this model should run for before exiting.
static uint32_t simulation_ticks = 0;

static inline accum get_random_prob()
{
   uint32_t random_num;
   random_num = sark_rand()&0x7FFF;
   return (accum)random_num >> 15;
}

static inline int get_min_block_bits(num)
{
   float needed_bits = log2(num);
   return ceil(needed_bits);
}

static inline int get_power_of_2_block_sz(num)
{
   int   next_power_of_2 = get_min_block_bits(num);
   return pow(2, next_power_of_2);
}

static inline void add_score_up_event()
{
  spin1_send_mc_packet(key | SPECIAL_EVENT_SCORE_UP, 0, NO_PAYLOAD);
  log_debug("Score up");
}

static bool initialize(uint32_t *timer_period)
{
  uint8_t * pMyScreen;
  uint no_action_index, left_action_index, right_action_index;
  accum no_action_rnd, left_action_rnd, right_action_rnd, sum;
  accum normed_no_action_rnd, normed_left_action_rnd, normed_right_action_rnd;
  int j;
  scale = pow(2, scale_shift);
  cum_score_up   = 0;
  cum_score_down = 0;
  log_info("Initialise of rlcontroller : started now!");
  sark_srand(5);

  // Get the address at which this core's DTCM data starts, from SRAM
  address_t address = data_specification_get_data_address();

  // Read the header
  if (!data_specification_read_header(address))
  {
      return false;
  }

  // Get the timing details and set up the simulation interface
  if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL, data_specification_get_region(REGION_PROVENANCE, address)))
  {
      return false;
  }

  // Read breakout region
  address_t breakout_region = data_specification_get_region(REGION_BREAKOUT, address);
  key = breakout_region[0];
  log_info("\tKey=%08x", key);

  log_info("Initialise: completed successfully");

  // Reserve memory for game state tracking:
  sz = GAME_HEIGHT/scale * GAME_WIDTH/scale;
  pMyScreen = sark_alloc((int)sz, sizeof(uint8_t));
  pScreen[0] = pMyScreen;
  if (!pMyScreen) {
     for (j=0; j<sz; j++)
        *(pMyScreen + j) = (uint8_t) 0;
  }

  // Reserve memory in SDRAM for V and Q arrays:
  bits_for_width  = get_min_block_bits(GAME_WIDTH/scale);
  bits_for_height = get_min_block_bits(GAME_HEIGHT/scale);
  bat_x_bit_start = 0;
  ball_y_bit_start = bat_x_bit_start  + bits_for_width;
  ball_x_bit_start = ball_y_bit_start + bits_for_height;
  action_bit_start = ball_x_bit_start + bits_for_width;
  value_statespace_elements = get_power_of_2_block_sz(GAME_WIDTH/scale) * 
                              get_power_of_2_block_sz(GAME_HEIGHT/scale) * 
                              get_power_of_2_block_sz(GAME_WIDTH/scale); // 16 * 8 * 16
  log_info("Width: %d, height: %d", (GAME_WIDTH/scale), (GAME_HEIGHT/scale));
  log_info("Scale shift: %d   scale: %d", scale_shift, scale);
  log_info("Screen size: %d x %d = %d", GAME_WIDTH/scale, GAME_HEIGHT/scale, sz);
  pVbase    = sark_xalloc(sv->sdram_heap, value_statespace_elements * 4, 0, ALLOC_LOCK);
  pEligbase = sark_xalloc(sv->sdram_heap, value_statespace_elements * 4, 0, ALLOC_LOCK);
  for (j=0; j<value_statespace_elements; j++) {
      *(pVbase+j)    = sark_rand()>>24;
      *(pEligbase+j) = (uint32_t) 0;
  }

  elements_per_action = value_statespace_elements;
  action_elements = elements_per_action * NUM_ACTIONS;
  log_info("Num actions: %d", NUM_ACTIONS);
  log_info("Policy space: %d", action_elements);
  log_info("Elements per action: %d", elements_per_action);
  pQbase  = sark_xalloc(sv->sdram_heap, action_elements * sizeof(uint32_t), 0, ALLOC_LOCK);
  for (j=0; j<elements_per_action; j++) {
      no_action_rnd    = 1.0; // + get_random_prob()/4;
      left_action_rnd  = 1.0; // + get_random_prob()/4;
      right_action_rnd = 1.0; // + get_random_prob()/4;
      sum = left_action_rnd + right_action_rnd + no_action_rnd;
      normed_no_action_rnd = no_action_rnd;///sum;
      normed_left_action_rnd = left_action_rnd;///sum;
      normed_right_action_rnd = right_action_rnd;///sum;

      *(pQbase+0*elements_per_action+j) = normed_left_action_rnd;
      *(pQbase+1*elements_per_action+j) = normed_right_action_rnd;
      *(pQbase+2*elements_per_action+j) = normed_no_action_rnd;
      *(pQbase+3*elements_per_action+j) = 0;
  }

  return true;
}

//----------------------------------------------------------------------------
// Callbacks
//----------------------------------------------------------------------------

void timer_callback(uint unused, uint dummy)
{
   uint32_t j;
   accum random_policy_choice;
   int chose = 0;
   accum random_num, denom;
   accum delta, frac_success;
   accum left_prob, right_prob, none_prob;
   uint action_selected;
   uint no_action_index, left_action_index, right_action_index;

   ticks ++;
   counter ++;

   if (!infinite_run && (ticks - 1) >= simulation_ticks)
   {
      //spin1_pause();
      // go into pause and resume state to avoid another tick
      simulation_handle_pause_resume(NULL);

      log_info("Exiting on timer.");
      return;
   }
   // Otherwise
   else
   {
      if (move_count > 0 && move_direction == KEY_LEFT) {
         spin1_send_mc_packet(key | KEY_LEFT, 0, NO_PAYLOAD);
         move_count--; 
      } else if (move_count > 0  && move_direction == KEY_RIGHT) {
         spin1_send_mc_packet(key | KEY_RIGHT, 0, NO_PAYLOAD);
         move_count--; 
      }

      // -----------------------------------
      // If we've moved, update the state functions:
      if ((ball_x != prev_ball_x) || (ball_y != prev_ball_y) || (bat_x != prev_bat_x)) {
         total_move_count++;
         // what score change has happened since the last state change:
         cum_score_up += score_up;
         cum_score_down += score_down;
         recent_reward_or_punishment = 3*score_up - score_down;
         score_up   = 0;
         score_down = 0;

         // We're in a new state. Update V & Q fields:
         // Construct state vectors and retrieve current value:
         prev_state_index = ((prev_ball_x>>scale_shift)<<ball_x_bit_start) + 
                            ((prev_ball_y>>scale_shift)<<ball_y_bit_start) + 
                            ((prev_bat_x>>bat_x_bit_start));
         prev_v_value = *(pVbase + prev_state_index);
         prev_q_index = ((prev_ball_x>>scale_shift)<<ball_x_bit_start) + 
                        ((prev_ball_y>>scale_shift)<<ball_y_bit_start) + 
                        ((prev_bat_x>>scale_shift)<<bat_x_bit_start)   +
                        prev_action*elements_per_action;

         state_index = ((ball_x>>scale_shift)<<ball_x_bit_start) + 
                       ((ball_y>>scale_shift)<<ball_y_bit_start) + 
                       ((bat_x>>scale_shift)<<bat_x_bit_start);
         v_value = *(pVbase + state_index);

         if (ball_x < 0 || ball_x > 159 || ball_y < 0 || ball_y > 159 || ball_x < 0 || ball_x > 159) {
            log_info("Out of range!");
         }

         // -----------------------------------
         // Select new action:
         // Get Pi values for each possible action:
         left_action_index  = (0<<action_bit_start) + ((ball_x>>scale_shift)<<ball_x_bit_start) + 
                                                      ((ball_y>>scale_shift)<<ball_y_bit_start) + 
                                                      ((bat_x>>scale_shift)<<bat_x_bit_start);
         right_action_index = (1<<action_bit_start) + left_action_index;
         no_action_index    = (1<<action_bit_start) + right_action_index;
         q_left  = *(pQbase + left_action_index);
         q_right = *(pQbase + right_action_index);
         q_none  = *(pQbase + no_action_index);

         // ***** Softmax function ******
         // This *is* required, as the current scheme fails when any of the three variables goes negative,
         // it just never gets picked again!
         // So we need to add an Exp function.
         //denom = exp(q_left) + exp(q_right) + exp(q_none);
         //left_prob = exp(q_left)/denom;
         //right_prob = exp(q_right)/denom;
         //none_prob = exp(q_none)/denom;
         // *****************************

         // Linearised probability model (no exp, but doesn't work well!)
         denom = q_left + q_right + q_none;
         left_prob = q_left/denom;
         right_prob = q_right/denom;
         none_prob = q_none/denom;

         // Chose the action based on soft-max of their Q values:
         if (random_policy_choice < left_prob) {
            move_direction = KEY_LEFT;
            left_total++;
         }
         else if (random_policy_choice < (left_prob+right_prob)) {
            move_direction = KEY_RIGHT;
            right_total++;
         }
         else { 
            move_direction = KEY_NONE;
            none_total++;
         }
         move_count = 40+(int)(get_random_prob()*100);
         current_action = move_direction - 1; 

         // -----------------------------------
         // Update Q value using SARSA rule:
         // Old index and Q value:
         prev_q_index = ((prev_ball_x>>scale_shift)<<ball_x_bit_start) + 
                        ((prev_ball_y>>scale_shift)<<ball_y_bit_start) + 
                        ((prev_bat_x>>scale_shift)<<bat_x_bit_start) + 
                          prev_action*elements_per_action;
         prev_q_value = *(pQbase + prev_q_index);
         // New index and q-value:
         q_index = ((ball_x>>scale_shift)<<ball_x_bit_start) + 
                   ((ball_y>>scale_shift)<<ball_y_bit_start) + 
                   ((bat_x>>scale_shift)<<bat_x_bit_start)   + 
                     current_action*elements_per_action;
         q_value = *(pQbase + q_index);

         // Update old q-value (SARSA rule):
         q_increment = alpha * (recent_reward_or_punishment + discount_rate * q_value - prev_q_value);
         *(pQbase + prev_q_index) = *(pQbase + prev_q_index) + q_increment;

         // -----------------------------------
         // Copy new state to the previous one:
         prev_ball_x = ball_x;
         prev_ball_y = ball_y;
         prev_bat_x  = bat_x;
         prev_action = current_action;
      }
   }
   outcount ++;
   if (outcount > 10000) {
      outcount = 0;
      log_info("Mvcount: %d, Move: %d, Ball: %d:%d  Bat: %d   L: %d, R:%d, N:%d", 
                total_move_count, move_direction, ball_x, ball_y, bat_x, 
                left_total, right_total, none_total);
      log_info("p-left: %k, p-right: %k, p-none: %k,  Q-inc: %k", left_prob, right_prob, none_prob, q_increment);
   }
}

// Process incoming packets, either concrning screen info (bat or ball position) or 
// reward/penalty info.
void mc_packet_received_callback(uint key, uint payload)
{
  use(payload);
  //log_info("Packet received %08x", key);
  new_key = key & 0xFFFFF;
  //if (((key >> 17) & 0x3) == SPECIAL_EVENT_MAX)
  if (new_key  >= SPECIAL_EVENT_MAX)
  { 
     color_bit = new_key & 0x1;
     // Only track solid objects, not background:
     if (color_bit) {
        change_y = (new_key >> 1) & 0xFF;
        change_x = (new_key >> 9) & 0xFF;

        // update bat and ball co-ordinates:
        if (change_y == (GAME_HEIGHT -1)) {
           // movement is the bat:
           bat_x = change_x>>scale_shift;
           if (bat_x <0)
              bat_x = 0;
           if (bat_x > 9)
              bat_x = 9;
        }
        else {
           // Change is the ball:
           // Update ball position:
           ball_y = change_y>>scale_shift;
           ball_x = change_x>>scale_shift;
           if (ball_x <0)
              ball_x = 0;
           if (ball_x > 9)
              ball_x = 9;
           if (ball_y <0)
              ball_y = 0;
           if (ball_y > 7)
              ball_y = 7;
        }
     }
  }
  else
  {
    // Reward/punishment:
    if (new_key == SPECIAL_EVENT_SCORE_UP){
        score_up++;
        //log_info("+");
    }
    else if (new_key == SPECIAL_EVENT_SCORE_DOWN) {
        score_down++;
        //log_info("-");
    }
  }  
  //(SPECIAL_EVENT_MAX + (i << 9) + (j << 1) + colour_bit);


/* SD: Insert stuff here */
}

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  int i,j;
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    log_error("Error in initialisation - exiting!");
    rt_error(RTE_SWERR);
    return;
  }

  /*init_frame();
  keystate = 0; // IDLE
  tick_in_frame = 0;
*/
  // Set timer tick (in microseconds)
  spin1_set_timer_tick(1000);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  ticks = 0;
  counter = 0;
  color_bit = -1;
  i = -1;
  j = -1;
  score_up   = 0;
  score_down = 0;
  move_count = 0;
  move_direction = KEY_LEFT;
  ball_x = 1;
  ball_y = 8;
  bat_x =  1;
  outcount = 0;

  simulation_run();

  // Free the screen memory:
  for(i=0; i<numGrids; i++) {
     sark_free(pScreen[i]);
  }
  
}

/*
         // Calculate the new error (delta):
         delta = recent_reward_or_punishment + discount_rate * v_value - prev_v_value;
         log_info("Val: %k reward: %k, delta: %k", v_value, recent_reward_or_punishment, delta);

         // Update the eligability value for this new state as we've visited it:
         *(pEligbase+state_index) = *(pEligbase+state_index) + 1.0;

         // Update all V values with this error (delta) based on their eligability:
         for (j=0; j<value_statespace_elements; j++) {
            *(pVbase+j) += alpha * delta * *(pEligbase+j);
            *(pEligbase+j) = discount_rate * lambda * *(pEligbase+j);
         }
*/

/*
      // Every few ticks, make an action:
      if (counter >= TICKS_PER_RESPONSE)
      {
         // Construct state vectors and retrieve current value:
         state_index = ((ball_x>>4)<<7) + ((ball_y>>4)<<4) + ((bat_x>>4));
         v_value = *(pVbase + state_index);
         // Get Pi values for each possible action:
         left_action_index  = (0<<11) + ((ball_x>>4)<<7) + ((ball_y>>4)<<4) + ((bat_x>>4));
         right_action_index = (1<<11) + ((ball_x>>4)<<7) + ((ball_y>>4)<<4) + ((bat_x>>4));
         no_action_index    = (2<<11) + ((ball_x>>4)<<7) + ((ball_y>>4)<<4) + ((bat_x>>4));
         pi_left  = *(pQbase + left_action_index);
         pi_right = *(pQbase + right_action_index);
         pi_none  = *(pQbase + no_action_index);
         //pi_left  = *(pQbase + 10);
         //pi_right = *(pQbase + 110);
         //pi_none  = *(pQbase + 200);
         //log_info("BallX: %d  BallY: %d  BatX: %d", ball_x, ball_y, bat_x);

         // Choose an action:
         move_count = 40; // We'll do the selected action twenty times.
         random_uint32 = sark_rand()>>16;
         random_num = (accum)(random_uint32)/0xFFFF;
         if (pi_left > random_num) {
            // Go left
            move_direction = KEY_LEFT;
            //log_info("Left");
         } else if ((pi_left + pi_right) > random_num) {
            // Go right
            move_direction = KEY_RIGHT;
            //log_info("Right");
         } else {
            // Don't move
            move_direction = KEY_NONE;
            //log_info("None");
         }
         counter = 0;
         //outcount ++;
         //if (outcount > 40) {
         //    outcount = 0;
         //    log_info("Random accum: %k, direction: %d ", random_num, move_direction);
         //}
      }
*/
