//
//  Re-enforcement learning for BreakOut
//
//
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
//#include <stdfix-exp.h>

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
#define ROUNDED_NUM_ACTIONS 4
#define NUM_ACTIONS 3

#define numGrids 1

#define TICKS_PER_RESPONSE 14

#define ELIG_TRACELIST_SZ 20

// **** Change scale shift to change the coarseness of the state space ****
// Image (and hence each state dimension) is scaled by 2 ** scale_shift.
int      scale_shift = 3;
int      scale;
accum    discount_rate = 0.9;
accum    score_weighting = 0.1;
accum    alpha  = 0.005;
accum    elig_decay = 0.95;
accum    scale_factor;
static uint32_t key;
uint32_t ticks;
uint32_t score_up, score_down; 
uint32_t new_key;
uint32_t color_bit;
uint32_t sz;
int32_t  new_y, new_x, change_x;
int32_t  prev_ball_y, prev_ball_x;
int32_t  prev_bat_x, bat_x_fine;
int32_t  ball_y, ball_x;
int32_t  bat_x, bat_x_fine;
int32_t  bat_x_segment;
int32_t  max_bat_x, max_ball_x, max_ball_y;
uint32_t * pEligbase;
accum *  pQbase;
accum    q_value, prev_q_value;
accum    delta;
accum    q_left, q_right, q_none;
accum    left_prob, right_prob, none_prob;
uint32_t q_index, prev_q_index;
uint32_t state_index;
uint32_t current_action = KEY_NONE;
uint32_t prev_action    = KEY_NONE;
uint32_t move_count;
uint32_t greedy_move_count;
uint32_t outcount       = 0;
uint32_t eligcount      = 0;
uint32_t value_statespace_elements;
uint32_t action_elements, elements_per_action;
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
accum    prob_greedy_action = 0.5;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

//! the number of timer ticks that this model should run for before exiting.
static uint32_t simulation_ticks = 0;

static accum get_random_prob()
{
   uint32_t random_num;
   random_num = sark_rand()&0x7FFF;
   return (accum)random_num >> 15;
}

static int get_min_block_bits(num)
{
   float needed_bits = log2(num);
   return ceil(needed_bits);
}

static int get_power_of_2_block_sz(num)
{
   int   next_power_of_2 = get_min_block_bits(num);
   return pow(2, next_power_of_2);
}

static void add_score_up_event()
{
  spin1_send_mc_packet(key | SPECIAL_EVENT_SCORE_UP, 0, NO_PAYLOAD);
  log_debug("Score up");
}

uint32_t create_index(int32_t bllx, int32_t blly, int32_t batx, int32_t action, int temp)
{
   uint32_t scrIdx = (bllx<<ball_x_bit_start) + (blly<<ball_y_bit_start) + 
                     (batx<<bat_x_bit_start);
   uint32_t actIdx = action * (1<<action_bit_start);

   //log_info("%d : %d : %d - %d  = %d (%d)", bllx, blly, batx, action, actIdx + scrIdx, temp);
   return actIdx + scrIdx;
}

accum get_q_value_from_state(bllx, blly, batx, a)
{
   // Create the index associated with the screen state (ball, bat)
   // and that coming from the given action number:
   uint32_t index = create_index(bllx, blly, batx, a, 1);

   return *(pQbase + index);
}

accum get_q_value(uint32_t index)
{
   return *(pQbase + index);
}

void set_q_value(uint32_t index, accum newVal)
{
   *(pQbase + index) = newVal;
}


static bool initialize(uint32_t *timer_period)
{
   accum no_action_rnd, left_action_rnd, right_action_rnd, sum;
   accum normed_no_action_rnd, normed_left_action_rnd, normed_right_action_rnd;
   int a, j;
   scale = pow(2, scale_shift);
   //log_info("Initialise of rlcontroller : started now!");
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
   //log_info("\tKey=%08x", key);
 
   //log_info("Initialise: completed successfully");
 
   // Reserve memory for game state tracking:
   sz = GAME_HEIGHT/scale * GAME_WIDTH/scale;
 
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
 
   elements_per_action = value_statespace_elements;
   action_elements = elements_per_action * ROUNDED_NUM_ACTIONS;
   pQbase        = sark_xalloc(sv->sdram_heap, action_elements * sizeof(uint32_t), 0, ALLOC_LOCK);
   pEligbase     = sark_alloc(ELIG_TRACELIST_SZ, sizeof(uint32_t));
   for (j=0; j<elements_per_action; j++) {
       *(pQbase+0*elements_per_action+j)    = 1.0k + get_random_prob();
       *(pQbase+1*elements_per_action+j)    = 1.0k + get_random_prob();
       *(pQbase+2*elements_per_action+j)    = 1.0k + get_random_prob();
       *(pQbase+3*elements_per_action+j)    = 0;
   }
   for(j=0; j<ELIG_TRACELIST_SZ; j++)
   {
      *(pEligbase+j)= -1;
   }
   max_bat_x  = GAME_WIDTH/scale - 1;
   max_ball_x = GAME_WIDTH/scale - 1;
   max_ball_y = GAME_HEIGHT/scale - 1;
   log_info("Q-value base address: 0x%x,  size (bytes)", pQbase, elements_per_action*12);
   //log_info("Width: %d, height: %d", (GAME_WIDTH/scale), (GAME_HEIGHT/scale));
   //log_info("Scale shift: %d   scale: %d", scale_shift, scale);
   //log_info("Screen size: %d x %d = %d", GAME_WIDTH/scale, GAME_HEIGHT/scale, sz);
   //log_info("Num actions: %d", NUM_ACTIONS);
   //log_info("Elements per action: %d", elements_per_action);
   return true;
}

bool check_for_game_state_change() {
   if ((ball_x != prev_ball_x) || (ball_y != prev_ball_y) || (bat_x != prev_bat_x))
   {
      prev_ball_x = ball_x;
      prev_ball_y = ball_y;
      prev_bat_x  = bat_x;
      return true;
   }
   else 
      return false;
}

void select_action()
{
   accum random_action_choice, are_we_doing_greedy_action;
   accum denom;
   uint32_t action_index;

   total_move_count++;

   q_left  = get_q_value_from_state(ball_x, ball_y, bat_x, 0); // Left action
   q_right = get_q_value_from_state(ball_x, ball_y, bat_x, 1); // Right action
   q_none  = get_q_value_from_state(ball_x, ball_y, bat_x, 2); // None action

   denom = q_left + q_right + q_none;
   left_prob = q_left/denom;
   right_prob = q_right/denom;
   none_prob = q_none/denom;

   are_we_doing_greedy_action = get_random_prob();
   // Are we doing a random action or just picking whichever has highest q?
   if (are_we_doing_greedy_action < prob_greedy_action)
   {
      greedy_move_count++;
      move_count     = 60;
      if ((left_prob > right_prob) && (left_prob > none_prob))
      {
         move_direction = KEY_LEFT;
         q_value = q_left;
         left_total++;
      }
      else if ((right_prob > left_prob) && (right_prob > none_prob))
      {
         move_direction = KEY_RIGHT;
         q_value = q_right;
         right_total++;
      }
      else // No action is currently best!
      {
         move_direction = KEY_NONE;
         q_value = q_none;
         none_total++;
      }
   }
   else  // Do random action:
   {
      move_count     = (int)(get_random_prob()*80 + 40);
      random_action_choice = get_random_prob();
      if (random_action_choice < 0.4) {
         move_direction = KEY_LEFT;
         q_value = q_left;
         left_total++;
      }
      else if (random_action_choice < 0.8) {
         move_direction = KEY_RIGHT;
         q_value = q_right;
         right_total++;
      }
      else { 
         move_direction = KEY_NONE;
         q_value = q_none;
         none_total++;
      }
   }
   action_index = move_direction - 1;
   q_index = create_index(ball_x, ball_y, bat_x, action_index, 2);
}

accum calculate_error_value()
{
   accum score = 10 * score_up - score_down;
   score_up    = 0;
   score_down  = 0;

   return score_weighting * score + discount_rate * prev_q_value - q_value;
}

void advance_eligibility_list()
{
   int i;
   for (i = ELIG_TRACELIST_SZ; i > 0; i--)
   {
     *(pEligbase + i) = *(pEligbase + i-1); 
   }
}

// Update the elgibility trace list for the state we just recently:
void add_latest_visited_state_to_list()
{
   // First advance the list by one, th create room for the new entry:
   advance_eligibility_list();

   // Now add the index of the new entry:
   *(pEligbase) = q_index;
}

// Run through the list of recent visited (s,a) pairs and update their
// Q values based on the latest delta:
void perform_reinforcement_learning()
{
   int i;
   uint32_t qa_state_index;
   accum this_q_value, new_q_value;
   accum elig_discount = 1;
   accum change;
   scale_factor = alpha * delta;

   for (i=1; i<ELIG_TRACELIST_SZ;i++)
   {
      qa_state_index = *(pEligbase + i);
      this_q_value = get_q_value(qa_state_index);
      // Add scaled value based on the current error, delta
      // and the decay of the elig trace:
      change = scale_factor * elig_discount;
      new_q_value = this_q_value + change;
      if (new_q_value < 0.05)
         new_q_value = 0.05;
      //if (outcount == 4000)
      //   log_info("Inc: %k", change);
      // update the q value for this state-action pair:
      set_q_value(qa_state_index, new_q_value);
      
      elig_discount = elig_discount * elig_decay;
   }
}

void next_state()
{
   prev_q_value = q_value;
   prev_q_index = q_index;
}

// XCV
void choose_next_action_and_update_rl_state()
{
   /*int i;
   eligcount++;
   if (eligcount == 1000 || eligcount == 1001)
   { 
      log_info("Elig: %d", eligcount); 
      for (i=0; i<ELIG_TRACELIST_SZ;i++)
      {
        log_info("E:%d", *(pEligbase + i)); 
      }
   }  */
   // Choose action:
   select_action();
   // Calculate TD error, delta:
   delta = calculate_error_value();
   // Update trace for last state:
   add_latest_visited_state_to_list();
   // Do reinforcement learning across recently used state-action pairs:
   perform_reinforcement_learning();
   // update q value state to point to the newly select state-action pair:
   next_state();
}


//----------------------------------------------------------------------------
// Callbacks
//----------------------------------------------------------------------------

void timer_callback(uint unused, uint dummy)
{
   ticks ++;

   if (!infinite_run && (ticks - 1) >= simulation_ticks)
   {
      //spin1_pause();
      // go into pause and resume state to avoid another tick
      simulation_handle_pause_resume(NULL);

      //log_info("Exiting on timer.");
      return;
   }

   // Perform action is we have one ongoing:
   if (move_count > 0 && move_direction == KEY_LEFT) {
      spin1_send_mc_packet(key | KEY_LEFT, 0, NO_PAYLOAD);
      move_count--; 
   } else if (move_count > 0  && move_direction == KEY_RIGHT) {
      spin1_send_mc_packet(key | KEY_RIGHT, 0, NO_PAYLOAD);
      move_count--; 
   }

   outcount ++;
   if (outcount > 20000) {
      outcount = 0;
      if (prob_greedy_action < 0.9)
         prob_greedy_action += 0.0001;

      //log_info("Mvcount: %d, Move: %d, Ball: %d:%d  Bat: %d (%d, %d)  L: %d, R:%d, N:%d", 
      //          total_move_count, move_direction, ball_x, ball_y, bat_x, bat_x_fine, bat_x_segment,
      //          left_total, right_total, none_total);
      //log_info("Scale factor: %k", scale_factor);
      //log_info("p-greed: %k", prob_greedy_action);
      //log_info("p-left: %k, p-right: %k, p-none: %k,  Delta: %k p-greed: %k", left_prob, right_prob, none_prob, delta, prob_greedy_action);
      log_info("q-left: %k, q-right: %k, q-none: %k", q_left, q_right, q_none);
      log_info("tot mov: %d  greedy moves: %d (greedy prob: %k)", total_move_count, greedy_move_count, prob_greedy_action);
   }
}


// Process incoming packets, either concrning screen info (bat or ball position) or 
// reward/penalty info.
void mc_packet_received_callback(uint key, uint payload)
{
   use(payload);
   new_key = key & 0xFFFFF;
   if (new_key  >= SPECIAL_EVENT_MAX)
   {
      color_bit = new_key & 0x1;
      // Only track solid objects, not background:
      if (color_bit == 1)
      {
         new_y = ((new_key >> 1) & 0xFF) - 1;
         new_x = (new_key >> 9) & 0xFF;
 
         //==============================================
         // update bat and ball co-ordinates:
         if (new_y == (GAME_HEIGHT -1)) {
            // Movement is probably the bat:
            bat_x_fine = new_x;
            if (bat_x_fine < 0)
               bat_x = 0;
            else if (bat_x_fine > max_bat_x<<scale_shift)
               bat_x = max_bat_x;
            else
               bat_x = bat_x_fine>>scale_shift;
         }
         else {
            // Change is the ball:
            // Update ball position:
            ball_y = new_y>>scale_shift;
            ball_x = new_x>>scale_shift;
            if (ball_x <0)
               ball_x = 0;
            if (ball_x > max_ball_x)
               ball_x = max_ball_x;
            if (ball_y <0)
               ball_y = 0;
            if (ball_y > max_ball_y)
               ball_y = max_ball_y;
         }
         //==============================================
      }
      // If the game state has changed, update the RL state:
      if (check_for_game_state_change())
         choose_next_action_and_update_rl_state();
      }
   else // Reward/punishment:
   {
      if (new_key == SPECIAL_EVENT_SCORE_UP){
         score_up++;
      }
      else if (new_key == SPECIAL_EVENT_SCORE_DOWN) {
         score_down++;
      }
   }  
}

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    //log_error("Error in initialisation - exiting!");
    rt_error(RTE_SWERR);
    return;
  }

  /*init_frame();
  keystate = 0; // IDLE
  tick_in_frame = 0;
*/
  // Set timer tick (in microseconds)
  spin1_set_timer_tick(timer_period);
  //log_info("Timer period: %d", timer_period);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  ticks = 0;
  color_bit = -1;
  score_up   = 0;
  score_down = 0;
  move_count = 0;
  greedy_move_count = 0;
  move_direction = KEY_LEFT;
  ball_x = 1;
  ball_y = 8;
  bat_x =  1;
  outcount = 0;
  eligcount=0;

  simulation_run();
}

