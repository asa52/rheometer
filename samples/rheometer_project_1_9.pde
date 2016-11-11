import java.util.Arrays;
import processing.serial.*;

String portName;
Serial serialPort;
int speed = 115200;
int bytesReceived = 0;
int c1, c2, c3, i, j, amp, trough, centre, peak, peak_to_peak; 
int A0_period_estimate, phase_estimate, sym_check;
int t_d = 0, t_a = 0, t = 0, interval = 160, val, pos_d, pos_a, x_offset = 60;
boolean t_sent = false;
int y_half_axis_pixel = 128;
int top_strain_pixel = 150;
int top_stress_pixel = top_strain_pixel + 2 * y_half_axis_pixel + 40;
int[] DAC1 = new int[1024], A0 = new int[1024];;//c = new int[1024];
float freq_val; int strain_val, stress_val, func_val, stress_rate = 120;
int k_val, b_val, eq_val;
byte[] val_in = new byte[5];
long last_time, time_difference, time_t_0;
long[] dt_ns = new long[1024];
int[] dt_ms = new int[1024];

PrintWriter output;

class menu_button {
  int x_0, y_0, x_width = 128, y_width = 26;
  String button_text = "";
  int button_text_height = 12;
  boolean button_activated = false;
  //int text_x_0 = (x_0 + x_width/2), text_y_0 = (y_0 + y_width/2 + button_text_height/2) ; 
  int opagueness = 64;
  color button_colour = color(150,150,150), text_colour = color(255,255,255); 
 
  void draw_button(){
   stroke(50);
   if(check_mouse_on_button()){button_colour = color(200,200,200);}
   else {button_colour = color(150,150,150);}
   fill(button_colour,opagueness);
   rect(x_0, y_0, x_width, y_width);
   if(button_activated){text_colour = color(0,255,0);}
   else {text_colour = color(255,0,0);}
   fill(text_colour);
   textSize(button_text_height);
   textAlign(CENTER);
   text(button_text, (x_0 + x_width/2), (y_0 + y_width/2 + button_text_height/2));
   textSize(10);
   textAlign(LEFT);
   fill(0);
  }
  
  boolean check_mouse_on_button(){
   if( x_0 <= mouseX && (x_0 + x_width) >= mouseX 
   && y_0 <= mouseY && (y_0 + y_width) >= mouseY){return true;} 
   else{return false;}       
  }
   
}
menu_button wfile = new menu_button();
menu_button fsweep = new menu_button();
menu_button oworcat = new menu_button();
menu_button epsig = new menu_button();
menu_button NRonoff = new menu_button();
menu_button NRview = new menu_button();
menu_button korb = new menu_button();
menu_button Torv = new menu_button();
menu_button ACDC = new menu_button();
menu_button eqcentre = new menu_button();
menu_button eqwrite = new menu_button();

class text_box {
  int x_0, y_0, x_width = 128, y_width = 26;
  String variable_text = "", constant_text = "";
  int box_text_height = 11, cursor_position = 0;
  boolean box_selected = false;
  color background_colour = color(0,0,0), text_colour = color(255,255,255); 
 
  void draw_box(){
   stroke(50);
   fill(background_colour);
   rect(x_0, y_0, x_width, y_width);   
   fill(text_colour);
   textSize(box_text_height);
   if(box_selected){
     int draw_cursor;
     if(0 <= cursor_position && variable_text.length() > cursor_position){
       draw_cursor = variable_text.length() - cursor_position;
     
    text("_", (x_0 + 2) 
      + (textWidth(variable_text.substring(0, draw_cursor))), 
         (y_0 + 1 + y_width/2 + box_text_height/2));
     }
      else if(cursor_position == variable_text.length()){
       text("_", (x_0 + 2), (y_0 + 1 + y_width/2 + box_text_height/2)); 
      }
  }
   textAlign(RIGHT);
   text(constant_text, (x_0 + x_width - 2), (y_0 + y_width/2 + box_text_height/2));
   textAlign(LEFT);
   text(variable_text, (x_0 + 2), (y_0 + y_width/2 + box_text_height/2));
   textSize(10);
   fill(0);
  }
  
  boolean check_mouse_on_box(){
   if( x_0 <= mouseX && (x_0 + x_width) >= mouseX 
   && y_0 <= mouseY && (y_0 + y_width) >= mouseY){return true;} 
   else{return false;}       
  }
   
}

text_box type_frequency = new text_box();
text_box type_set_strain = new text_box();
text_box type_file_name = new text_box();
text_box type_set_simu = new text_box();

// Variable to store text currently being typed
String typing = "", LHS = "", RHS = "";
int typing_cursor = 0;

// Variable to store saved text when return is hit
String saved = "", debug = "";
String file_name = "", set_strain_text = "2048", frequency_text = "5", set_stress_text = "";
String set_simu_k_text = "0", set_simu_b_text = "0", set_DC_text = "2047", set_centre_text = "";

void setup() {
  size(760, 740);
  frameRate(320);
  System.out.println("Available serial ports are: " + Arrays.toString( Serial.list())); //debug
  background(100);
  //Open 2nd visible serial port
  portName = Serial.list()[0];

  // Serial.list() array's indexes on my machine (this is machine specific):
  // 0 = integrated serial on motherboard
  // 1 = Arduino DUE programming port (when plugged in) - gets received by IDE, PuTTY, Processing
  // 2 = Arduino DUE native port (when plugged in also) - gets received by IDE, PuTTY, but NOT by Processing

  serialPort = new Serial(this, portName, speed);
  System.out.println( "Opened port " + portName); //debug
  
  strain_axis("microns", 150);
  stress_axis("nNm", 12);
  time_axes("ms", 1.67);
  initialise_interface();
}

void draw() {

      display_text();
      
      oscilloscope();
  
}

void serialEvent( Serial port) {
  while( port.available() > 0) {
 
    if(i == 0){
    c1 = port.read();
   }
   
    if(i == 1){
    c2 = port.read();
    
     if(64 > c2){
     thread("time_nano_to_micro");
     //time_nano_to_micro();
     DAC1[t_d] = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " " + c + " \n" ); //debug
     //System.out.print(DAC1[t_d] + " \n");
     t_d++;
     
     if((t_d % interval) == 0){
      pos_d = t_d;
      }
     
     if(t_d >= (width - 2*x_offset) ){
     t_d = 0;
     pos_d = t_d;
       
     }
     i = -1;
     t_sent = true;
    }
     
     if(64 <= c2 && 128 > c2){
     A0[t_a] = c1 + (c2-64)*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
     t_a++;
     
     if((t_a % interval) == 0){
      pos_a = t_a;
      }
     
     if(t_a >= (width - 2*x_offset) ){
     t_a = 0;
     pos_a = t_a;
       
     }
    }
    if(128 <= c2){
      
    c2 -= 128; 
    
    i = -1;
    }
           
  }
  
   if(i == 2){
    c3 = port.read();
    
    if(t_sent){t = c3;}
    
    else{
    if(0 == c3){ // none of the 3 byte's bits have other than signalling contant
                 // thus equality is sufficient for comparison
     amp = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    }
    
    if(1 == c3){
    sym_check = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    }
    
    if(2 == c3){
     centre = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    }
    
    if(4 == c3){
     peak_to_peak = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    }
   
    if(8 == c3){
     A0_period_estimate = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    }
   
    if(16 == c3){
     phase_estimate = c1 + c2*256;
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    } 
    
    if(17 == c3){
     phase_estimate = (c1 + (c2-128)*256);
     //System.out.print( c1 + " " + c2 + " pos = " + pos + " \n");
    } 
   }
    i = 1;
    t_sent =false;
   }
    
    
    i = 1 - i;
    bytesReceived++;
    }
}

void delay(int delay)
{
  int time = millis();
  while(millis() - time <= delay);
}

void keyPressed() {
  
  if(type_frequency.box_selected){
   typing_cursor = type_frequency.variable_text.length() - (type_frequency.cursor_position + 1);
   System.out.println("frequency cursor_position = " + type_frequency.cursor_position);
 }
  else if(type_set_strain.box_selected){
   typing_cursor = type_set_strain.variable_text.length() - (type_set_strain.cursor_position + 1);
   System.out.println("strain cursor_position = " + type_set_strain.cursor_position); 
 }
  else if(type_file_name.box_selected){
   typing_cursor = type_file_name.variable_text.length() - (type_file_name.cursor_position + 1);
   System.out.println("file cursor_position = " + type_file_name.cursor_position);
 }
  else if(type_set_simu.box_selected){
   typing_cursor = type_set_simu.variable_text.length() - (type_set_simu.cursor_position + 1);
   System.out.println("simu cursor_position = " + type_set_simu.cursor_position); 
 }
 
   System.out.println("typing_cursor = " + typing_cursor);   
  // If the return key is pressed, save the String and clear it
  if (key == '\n' ) {
   boolean send_ok = false; 
    saved = typing;

    typing = "";
    typing_cursor = -1;
    
    if(type_frequency.box_selected){
     freq_val = float(saved);
     float t_unit = (1000.0 / (freq_val*120.0) );    
     time_axes("ms", t_unit);
     if( 0.1 <= freq_val && 20 >= freq_val ){
     val = int( ( (1.0/freq_val) * 42000000.0 ) / 120.0);
     // needs to be HALF of (1/freq) * (84 MHz/sample_num) for some reason
     val_in[0] = 0;
     send_ok = true;
     type_frequency.constant_text = saved + " Hz";
     if(2.0 >= freq_val){interval = (width - 2*x_offset)/32;}
     else if(2.0 < freq_val && 6.0 >= freq_val){interval = (width - 2*x_offset)/8;}
     else if(6.0 < freq_val){interval = (width - 2*x_offset)/4;}
       }
     }
     
    else if(type_set_strain.box_selected && !ACDC.button_activated){
     if(!epsig.button_activated){  
     strain_val = int(saved);
     if( 100 <= strain_val && 4000 >= strain_val ){
     val = strain_val;  
     val_in[0] = 1;
     send_ok = true;
     type_set_strain.constant_text = saved + " mum/10";
     set_strain_text = saved;
       }
      }
     else if(epsig.button_activated){
     stress_val = int(saved);
     if( 0 <= stress_val && 2048 >= stress_val ){
     val = stress_val;
     val_in[0] = 1;
     send_ok = true;
     type_set_strain.constant_text = saved + " of 2048";
     set_stress_text = saved;
       }
      }
     }
     
     else if(type_set_strain.box_selected && ACDC.button_activated){  
     func_val = int(saved);
     if( 0 <= func_val && 4095 >= func_val ){
     val = func_val;  
     val_in[0] = 1;
     send_ok = true;
     type_set_strain.constant_text = saved + " of 4095";
     set_strain_text = saved;
       }
      }
     
     else if(type_set_simu.box_selected){
     if(!korb.button_activated && !eqwrite.button_activated){  
     k_val = int(saved);
     if( (-1024) <= k_val && 1024 >= k_val ){
     val = k_val;  
     val_in[0] = 3;
     send_ok = true;
     type_set_simu.constant_text = saved + " kDAC1eq";
     set_simu_k_text = saved;
       }
      }
     else if(korb.button_activated && !eqwrite.button_activated){
     b_val = int(saved);
     if( (-1024) <= b_val && 1024 >= b_val ){
     val = b_val;
     val_in[0] = 3;
     send_ok = true;
     type_set_simu.constant_text = saved + " bDAC1eq";
     set_simu_k_text = saved;
       }
      }
     else if(eqwrite.button_activated){
     eq_val = int(saved);
     if( 100 <= eq_val && 11000 >= eq_val ){
     val = eq_val;
     val_in[0] = 5;
     send_ok = true;  
     type_set_simu.constant_text = saved + " mum/10";
     set_centre_text = saved;
       }
      } 
     } 
     
   if(send_ok){  
     System.out.println( "val = " + val);
     //debug = saved;
     saved = "";
         
      
   val_in[4] = (byte) ((val >> 24) & 0xFF);
   val_in[3] = (byte) ((val >> 16) & 0xFF);
   val_in[2] = (byte) ((val >> 8) & 0xFF);
   val_in[1] = (byte) (val & 0xFF);
   
   
   serialPort.write(val_in);/*
   System.out.println( " val_in[0] = " + val_in[0]
                     + " val_in[1] = " + val_in[1]
                     + " val_in[2] = " + val_in[2]
                     + " val_in[3] = " + val_in[3]
                     + " val_in[4] = " + val_in[4]);*/ //debug

      }
      
   send_ok = false; 
  } 
  else if(key == '0' || key == '1' || key == '2' || key == '3' || key == '4' || key == '5' ||
          key == '5' || key == '6' || key == '7' || key == '8' || key == '9' || key == '.' ||
          (type_file_name.box_selected && !(key == 8 || key == 127 || key == CODED)) ||
          (type_set_simu.box_selected && key == '-') ){
    // Otherwise, concatenate the String
    // Each character typed by the user is added to the end of the String variable.
    //typing = typing + key;
    if( (typing.length() - 1) == typing_cursor){typing = typing + key;}
    else if( (typing.length() - 1) > typing_cursor){
      if(-1 == typing_cursor){
       LHS = "" + key;
       RHS = typing.substring(typing_cursor+1);
      }
      else{
      LHS = typing.substring(0,typing_cursor+1) + key; 
      RHS = typing.substring(typing_cursor+1);
      }
     typing = LHS + RHS;
    }
    
    typing_cursor ++;
   System.out.println("typing_cursor = " + typing_cursor);
   System.out.println("key was " + key); 
  }
  else if(-1 <= typing_cursor && (typing.length() - 1) >= typing_cursor){
   if(key == 127 && (typing.length() - 1) > typing_cursor){//ASCII 127 should be delete, 
    LHS = typing.substring(0,typing_cursor+1); 
    RHS = typing.substring(typing_cursor+2,typing.length());
    typing = LHS + RHS;   
   }
   if(key == 8){//ASCII 08 should be backspace
    if(0 >= typing_cursor){LHS = "";}
    else if(0 <= typing_cursor){LHS = typing.substring(0,typing_cursor);} 
    RHS = typing.substring(typing_cursor+1,typing.length());
    typing = LHS + RHS;
    if(-1 < typing_cursor){typing_cursor -= 1;}
   }
   System.out.println("typing_cursor = " + typing_cursor);
  }
  if(key == CODED){
    System.out.println("key == CODED");
   if(keyCode == LEFT && -1 < typing_cursor){
     typing_cursor -= 1;
     System.out.println("end of if LEFT typing_cursor = " + typing_cursor);
   }
   else if(keyCode == RIGHT && (typing.length() - 1) > typing_cursor){     
     typing_cursor += 1;
     System.out.println("end of if RIGHT typing_cursor = " + typing_cursor);
   }
  System.out.println("end of if CODED typing_cursor = " + typing_cursor); 
  }

 
 if(epsig.button_activated || ACDC.button_activated){ 
   if(key == 'w' && 2048 > amp){
   val_in[0] = 4;
   serialPort.write(val_in[0]);
   type_set_strain.constant_text = (amp + 1) + " of 2048";
   //delay(100); 
    
  }
  else if(key == 's' && 0 < amp){
   val_in[0] = 2;
   serialPort.write(val_in[0]);
   type_set_strain.constant_text = (amp - 1) + " of 2048"; 
   //delay(100); 
  }  
  }
  
      System.out.println("LHS = " + LHS);
      System.out.println("RHS = " + RHS);
System.out.println("typing_cursor = " + typing_cursor);
if(type_frequency.box_selected){
 type_frequency.variable_text = typing;
 type_frequency.cursor_position = typing.length() - typing_cursor - 1;
 System.out.println("frequnecy cursor_position = " + type_frequency.cursor_position);
 }
 else if(type_set_strain.box_selected){
 type_set_strain.variable_text = typing;
 type_set_strain.cursor_position = typing.length() - typing_cursor - 1;
 System.out.println("strain cursor_position = " + type_set_strain.cursor_position);
 }
    else if(type_file_name.box_selected){
    type_file_name.variable_text = typing;
    type_file_name.cursor_position = typing.length() - typing_cursor - 1;
    System.out.println("file cursor_position = " + type_file_name.cursor_position);
  }
  else if(type_set_simu.box_selected){
 type_set_simu.variable_text = typing;
 type_set_simu.cursor_position = typing.length() - typing_cursor - 1;
 System.out.println("simu cursor_position = " + type_set_simu.cursor_position);
 }

}


void mousePressed(){

            if(type_frequency.check_mouse_on_box() && !fsweep.button_activated){
              type_frequency.box_selected = !type_frequency.box_selected;
              typing = type_frequency.variable_text;
              type_set_strain.box_selected = false;
              type_file_name.box_selected = false;
              type_set_simu.box_selected = false;
            }
            else if(type_set_strain.check_mouse_on_box() && !fsweep.button_activated){
              type_set_strain.box_selected = !type_set_strain.box_selected;
              typing = type_set_strain.variable_text;
              type_frequency.box_selected = false;
              type_file_name.box_selected = false;
              type_set_simu.box_selected = false;
            }
            else if(type_file_name.check_mouse_on_box()){
              type_file_name.box_selected = !type_file_name.box_selected;
              typing = type_file_name.variable_text;
              type_frequency.box_selected = false;
              type_set_strain.box_selected = false;
              type_set_simu.box_selected = false;
            }
            else if(type_set_simu.check_mouse_on_box()){
              type_set_simu.box_selected = !type_set_simu.box_selected;
              typing = type_set_simu.variable_text;
              type_frequency.box_selected = false;
              type_file_name.box_selected = false;
              type_set_strain.box_selected = false;
            }
            else{
              type_set_strain.box_selected = false;
              type_frequency.box_selected = false;
              type_file_name.box_selected = false;
              type_set_simu.box_selected = false;
            }        
  
            if(oworcat.check_mouse_on_button()){
              oworcat.button_activated = !oworcat.button_activated;
              if(oworcat.button_activated){
              oworcat.button_text = "concatenate";
              }
              else{oworcat.button_text = "overwrite";}
            } 
  
  
            if(wfile.check_mouse_on_button()){
              wfile.button_activated = !wfile.button_activated;
              if(wfile.button_activated){
              thread("concatenate_file");
              wfile.button_text = "write file ON";
              }
              else{wfile.button_text = "write file OFF";}
            }
            
            /*
            if(fsweep.check_mouse_on_button()){
              fsweep.button_activated = !fsweep.button_activated;
              if(fsweep.button_activated){
                type_set_strain.box_selected = false;
                type_frequency.box_selected = false;
                if(!epsig.button_activated){
                thread("frequency_sweep");
                fsweep.button_text = "1-20 Hz sweep";
                }
                else if(epsig.button_activated){
                thread("stress_of_time");
                fsweep.button_text = "1 per cycle";
                }
              }
              else{fsweep.button_text = "manual mode";}
            }*/
            
            
            if(epsig.check_mouse_on_button() && !ACDC.button_activated){
              epsig.button_activated = !epsig.button_activated;
              type_set_strain.box_selected = false;
              type_frequency.box_selected = false;
              if(fsweep.button_activated){
              fsweep.button_activated = !fsweep.button_activated;
              fsweep.button_text = "manual mode";
            }
              if(epsig.button_activated){

                epsig.button_text = "stress";
                type_set_strain.constant_text = amp + " of 2048";
                //type_set_strain.constant_text = "of 2048";
                val_in[0] = 16;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
              else{
                epsig.button_text = "strain";
                type_set_strain.constant_text = set_strain_text + " mum/10";
                //type_set_strain.constant_text = "mum/10";
                val_in[0] = 8;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
             
            }
            
            if(ACDC.check_mouse_on_button()){
              ACDC.button_activated = !ACDC.button_activated;
              if(ACDC.button_activated){
              ACDC.button_text = "DC";
              type_set_strain.constant_text = set_DC_text + " of 4095";
              //val_in[0] = 128; //byte is signed in Processing
              //so can only have -127 <= byte <= +127
                val_in[0] = 7;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
              else{
                ACDC.button_text = "AC";
                if(epsig.button_activated){
                type_set_strain.constant_text = amp + " of 2048";
                val_in[0] = 16;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
                }
                else{
                type_set_strain.constant_text = set_strain_text + " mum/10";  
                val_in[0] = 8;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
                }              
             }
            }
            
            if(NRonoff.check_mouse_on_button()){
              NRonoff.button_activated = !NRonoff.button_activated;

              if(NRonoff.button_activated){
               if(NRview.button_activated){
               NRview.button_activated = false;
               NRview.button_text = "drive";
               }/*
               if(Torv.button_activated){
               Torv.button_activated = false;
               Torv.button_text = "T";
               }*/
                NRonoff.button_text = "NR on";
                val_in[0] = 64;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
              else{
                NRonoff.button_text = "NR off";
                val_in[0] = 32;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
             
            }
            
            if(NRview.check_mouse_on_button() && !NRonoff.button_activated){
              NRview.button_activated = !NRview.button_activated;

              if(NRview.button_activated){
                NRview.button_text = "view";
                val_in[0] = 12;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
              else{
                NRview.button_text = "drive";
                val_in[0] = 32;
                serialPort.write(val_in[0]);
                System.out.println("val_in[0] = " + val_in[0]);
              }
             
            }
            
            if(korb.check_mouse_on_button() && !eqwrite.button_activated){
              korb.button_activated = !korb.button_activated;
              val_in[0] = 6;
              serialPort.write(val_in[0]);
              System.out.println("val_in[0] = " + val_in[0]);
              if(korb.button_activated){
              korb.button_text = "b";
              type_set_simu.constant_text = set_simu_b_text + " bDAC1eq";
              }
              else{
              korb.button_text = "k";
              type_set_simu.constant_text = set_simu_k_text + " kDAC1eq";
              }
            }
            //NRview priotised over and freezes Torv
            if(Torv.check_mouse_on_button() 
               && !NRview.button_activated && !NRonoff.button_activated){
              Torv.button_activated = !Torv.button_activated;
              val_in[0] = 24;
              serialPort.write(val_in[0]);
              System.out.println("val_in[0] = " + val_in[0]);
              if(Torv.button_activated){Torv.button_text = "v";}
              else{Torv.button_text = "T";}
            }
  
            if(eqcentre.check_mouse_on_button()){
              eqcentre.button_activated = !eqcentre.button_activated;
              val_in[0] = 48;
              serialPort.write(val_in[0]);
              System.out.println("val_in[0] = " + val_in[0]);
              if(eqcentre.button_activated){eqcentre.button_text = "per";}
              else{eqcentre.button_text = "tem";}
            }
            
            if(eqwrite.check_mouse_on_button()){
              eqwrite.button_activated = !eqwrite.button_activated;
              if(eqwrite.button_activated){
              eqwrite.button_text = "1";
              type_set_simu.constant_text = set_centre_text + " mum/10";
            }
              else{
              eqwrite.button_text = "0";
              if(korb.button_activated){
              korb.button_text = "b";
              type_set_simu.constant_text = set_simu_b_text + " bDAC1eq";
              }
              else{
              korb.button_text = "k";
              type_set_simu.constant_text = set_simu_k_text + " kDAC1eq";
              }
              }
            }
}



void oscilloscope(){
   
  
    if((pos_a % interval) == 0){
   
   if( pos_a == 0){pos_a = (width - 2*x_offset);}
   for(j = (pos_a - interval); j <= (pos_a - 1); j++){
   stroke(255);  
   //point(j, 290 - (A0[j]-2048)/20);
   //point(j, 290 - ((A0[j]-5980)/50));
   point(x_offset + j, (top_strain_pixel + y_half_axis_pixel) - ((A0[j]-6000)/47));
   }
   if( pos_a == (width - 2*x_offset) ){pos_a = 0;}
   stroke(100);
   fill(100);
   rect(x_offset + pos_a, top_strain_pixel, interval, 2 * y_half_axis_pixel);
   
      }
   
   if((pos_d % interval) == 0){

   if( pos_d == 0){pos_d = (width - 2*x_offset);}
   for(j = (pos_d - interval); j <= (pos_d - 1); j++){
   stroke(255);  
   point(x_offset + j, (top_stress_pixel + y_half_axis_pixel) - ((DAC1[j]-2048)/16));
   //point(j, 530 - (DAC1[j]-5980)/60);
   //point(j, 290 - ((A0[j]-5980)/60));
   }
   if( pos_d == (width - 2*x_offset) ){pos_d = 0;}

   stroke(100);
   fill(100);
   rect(x_offset + pos_d, top_stress_pixel, interval, 2 * y_half_axis_pixel);
   //rect(pos_d, 180, interval, 240);
  }

  
}


void display_text(){
  
      textAlign(LEFT);
      textSize(10);
      stroke(100);
      fill(100);
      rect(0, 0, width, 120);
      fill(255);
      text("Reading: " + portName, 10, 20);
      text("Received " + bytesReceived + " bytes.", 10, 50);
      /*
      if(t_d == 0){
      text("DAC1 value " + DAC1[t_d] + " of 0 to 4095", 20, 80);
      }
      else{
      text("DAC1 value " + DAC1[t_d - 1] + " of 0 to 4095", 20, 80);
     }
      if(t_a == 0){
      text("mu value " + A0[t_a] + " in microns/10", 20, 110);
      }
      else{
      text("mu value " + A0[t_a - 1] + " in microns/10", 20, 110);  
      }*/
      text("DAC1 value " + DAC1[t_d] + " of 0 to 4095", 10, 80);
      text("mu value " + A0[t_a] + " in microns/10", 10, 110);
      
      text("amp value " + amp + " of 0 to 2048", 170, 20);
      //text("phase estimate " + (float(phase_estimate)/60.0) + " PI t = " + t, 200, 50);
      text("phase estimate " + phase_estimate + " steps, t = " + t, 170, 50);
      text("centre value " + centre + " in microns/10", 170, 80);
      text("A0 amp " + round(peak_to_peak/2) + " in microns/10", 170, 110);
      
      //text(typing,420,20);
      //text(saved,420,50);
      //text("DAC1 t diff " + dt_ns[1] + " in nanoseconds", 420, 80);
      text("Response period " + A0_period_estimate + " in t diff", 340, 80);
      text("DAC1 t diff " + dt_ms[1] + "   in microseconds", 340, 110);
      
      type_frequency.draw_box();
      type_set_strain.draw_box();
      type_file_name.draw_box();
      type_set_simu.draw_box();
      wfile.draw_button();
      //fsweep.draw_button();        
      oworcat.draw_button();
      epsig.draw_button();
      NRonoff.draw_button();
      NRview.draw_button();        
      korb.draw_button();
      Torv.draw_button();      
      ACDC.draw_button();
      eqcentre.draw_button();
      eqwrite.draw_button();     
  
}

void initialise_interface(){
  
  wfile.x_0 = 560;wfile.y_0 = 70;
  //fsweep.x_0 = 560;fsweep.y_0 = 100;
  oworcat.x_0 = 560;oworcat.y_0 = 40;
  epsig.x_0 = 470;epsig.y_0 = 40;epsig.x_width = 40;
  NRonoff.x_0 = 470;NRonoff.y_0 = 10;NRonoff.x_width = 42; 
  NRview.x_0 = 516;NRview.y_0 = 10;NRview.x_width = 36; 
  korb.x_0 = 690;korb.y_0 = 100;korb.x_width = 10;
  eqcentre.x_0 = 702;eqcentre.y_0 = 100;eqcentre.x_width = 30;
  eqwrite.x_0 = 732;eqwrite.y_0 = 100;eqwrite.x_width = 10; 
  Torv.x_0 = 690;Torv.y_0 = 40;Torv.x_width = 20;
  ACDC.x_0 = 514;ACDC.y_0 = 40;ACDC.x_width = 20;
  type_file_name.x_0 = 560;type_file_name.y_0 = 10;type_file_name.x_width = 160;
  type_frequency.x_0 = 340;type_frequency.y_0 = 10;
  type_set_strain.x_0 = 340;type_set_strain.y_0 = 40;
  type_set_simu.x_0 = 560;type_set_simu.y_0 = 100;
  type_frequency.constant_text = frequency_text + " Hz";
  type_set_strain.constant_text = set_strain_text + " mum/10";
  type_set_simu.constant_text = set_simu_k_text + " kDAC1eq";
  //type_frequency.variable_text = "5";
  //type_set_strain.variable_text = "2048";
  type_file_name.variable_text = "rheo_data_"+ day() + "_" + month() + "_" + year() + ".txt";
  wfile.button_text = "write file OFF";
  fsweep.button_text = "manual mode";
  oworcat.button_activated = true;
  oworcat.button_text = "concatenate";
  epsig.button_text = "strain";
  NRonoff.button_text = "NR off";
  NRview.button_text = "drive";
  korb.button_text = "k";
  Torv.button_text = "T";
  ACDC.button_text = "AC";
  eqcentre.button_text = "tem";
  eqwrite.button_text = "0";
}

int convert_DAC1_to_muAmp_pixels(int DAC1_value){
  float y_div = (2048 * 0.1623) / y_half_axis_pixel;
  int current_pixels = int( (float( DAC1_value - 2048 ) * 0.1623) / y_div );
  return current_pixels;
  
}

float convert_DAC1_to_muAmps(int DAC1_value){
  float current = (float( DAC1_value - 2048 ) * 0.1623);
  return current;
  
}

int convert_DAC1_to_nanoNm_pixels(int DAC1_value){
  float y_div = (2048 * 0.89845) / y_half_axis_pixel;
  int torque_pixels = int( (float( DAC1_value - 2048 ) * 0.89845) / y_div );
  return torque_pixels;
  
}

float convert_DAC1_to_nanoNm(int DAC1_value){
  float torque = (float( DAC1_value - 2048 ) * 0.89845);
  return torque;
  
}

void strain_axis(String strain_axis_label, int strain_unit){
  int strain_text_height = 11;
  int num_strain_div = 8;
  int strain_div_height = (2*y_half_axis_pixel)/num_strain_div;
  
  stroke(100);
  fill(100);
  rect(0, top_strain_pixel, x_offset - 5, 2 * y_half_axis_pixel);
  rect(x_offset - ((strain_text_height/2)*(strain_axis_label.length()/2)),
  top_strain_pixel - strain_text_height - 5, (strain_text_height/2)*strain_axis_label.length(),
  strain_text_height); 
  
  
  stroke(255);
  fill(255);
  line(x_offset - 2, top_strain_pixel, x_offset - 2, top_strain_pixel + 2 * y_half_axis_pixel);
  
  //point(x_offset - 3, top_strain_pixel + y_half_axis_pixel);
  //point(x_offset - 4, top_strain_pixel + y_half_axis_pixel);
  textSize(strain_text_height);
  textAlign(CENTER);
  text(strain_axis_label, x_offset , top_strain_pixel - strain_text_height/2);
  textAlign(RIGHT);
  //text(0, x_offset - 6, top_strain_pixel + y_half_axis_pixel);
  
  for(int i=0; i <= num_strain_div; i++){/*
    point(x_offset - 3, (top_strain_pixel + y_half_axis_pixel) + i * strain_div_height);
    point(x_offset - 4, (top_strain_pixel + y_half_axis_pixel) + i * strain_div_height);
    text((-1) * i * strain_unit, x_offset - 6, (top_strain_pixel + y_half_axis_pixel) + i * strain_div_height + strain_text_height/2);
    */    
    point(x_offset - 3, (top_strain_pixel + 2*y_half_axis_pixel) - i * strain_div_height);
    point(x_offset - 4, (top_strain_pixel + 2*y_half_axis_pixel) - i * strain_div_height);
    text(i * strain_unit, x_offset - 6, (top_strain_pixel + 2*y_half_axis_pixel) - i * strain_div_height + strain_text_height/2);

  }
  textAlign(LEFT);
}

void stress_axis(String stress_axis_label, int stress_unit){
  int stress_text_height = 10;
  int num_stress_div = 4;
  int stress_div_height = (y_half_axis_pixel)/num_stress_div;
  
  stroke(100);
  fill(100);
  rect(0, top_stress_pixel, x_offset - 5, 2 * y_half_axis_pixel);
  rect(x_offset - ((stress_text_height/2)*(stress_axis_label.length()/2)),
  top_stress_pixel - stress_text_height, (stress_text_height/2)*stress_axis_label.length(),
  stress_text_height); 
  
  stroke(255);
  fill(255);
  line(x_offset - 2, top_stress_pixel, x_offset - 2, top_stress_pixel + 2 * y_half_axis_pixel);
  
  point(x_offset - 3, top_stress_pixel + y_half_axis_pixel);
  point(x_offset - 4, top_stress_pixel + y_half_axis_pixel);
  textSize(stress_text_height);
  textAlign(CENTER);
  text(stress_axis_label, x_offset , top_stress_pixel - stress_text_height/2);
  textAlign(RIGHT);
  text(0, x_offset - 6, top_stress_pixel + y_half_axis_pixel);
  
  for(int i=1; i <= num_stress_div; i++){
    point(x_offset - 3, (top_stress_pixel + y_half_axis_pixel) + i * stress_div_height);
    point(x_offset - 4, (top_stress_pixel + y_half_axis_pixel) + i * stress_div_height);
    text((-1) * i * stress_unit, x_offset - 6, (top_stress_pixel + y_half_axis_pixel) + i * stress_div_height + stress_text_height/2);
    
    point(x_offset - 3, (top_stress_pixel + y_half_axis_pixel) - i * stress_div_height);
    point(x_offset - 4, (top_stress_pixel + y_half_axis_pixel) - i * stress_div_height); 
    text(i * stress_unit, x_offset - 6, (top_stress_pixel + y_half_axis_pixel) - i * stress_div_height + stress_text_height/2);

  }
  textAlign(LEFT);
}

void time_axes(String t_axis_label, float t_unit){
  int t_text_height = 10;
  int t_strain_y = top_strain_pixel + (2 * y_half_axis_pixel);
  int t_stress_y = top_stress_pixel + (2 * y_half_axis_pixel);
  int num_t_div = 16;
  int t_div_width = (width - 2*x_offset)/num_t_div;
  
  stroke(100);
  fill(100);
  rect(0, t_strain_y + 5, width , t_text_height);
  rect(0, t_stress_y + 5, width , t_text_height);
  rect(width - x_offset + 1, t_strain_y + t_text_height/2, (t_text_height/2)*t_axis_label.length(), t_text_height); 
  rect(width - x_offset + 1, t_stress_y + t_text_height/2, (t_text_height/2)*t_axis_label.length(), t_text_height);
  
  stroke(255);
  fill(255);
  line(x_offset, t_strain_y + 2, width - x_offset, t_strain_y + 2);
  line(x_offset, t_stress_y + 2, width - x_offset, t_stress_y + 2);
  textSize(t_text_height);
  textAlign(LEFT);
  text(t_axis_label, width - x_offset + 2, t_strain_y + t_text_height/2);
  text(t_axis_label, width - x_offset + 2, t_stress_y + t_text_height/2);
  textAlign(CENTER);
  
  for(int i=0; i <= num_t_div; i++){
    point(x_offset + i*t_div_width, t_strain_y + 3);
    point(x_offset + i*t_div_width, t_strain_y + 4);
    text(round(i * t_unit * t_div_width), x_offset + i*t_div_width, t_strain_y + 5 + t_text_height);
    
    point(x_offset + i*t_div_width, t_stress_y + 3);
    point(x_offset + i*t_div_width, t_stress_y + 4);
    text(round(i * t_unit * t_div_width), x_offset + i*t_div_width, t_stress_y + 5 + t_text_height);
    
    
  }
}

void time_nano_to_micro(){
  if(t_d > 0){
  time_difference = System.nanoTime() - last_time;
  dt_ns[t_d] = time_difference;
  last_time = System.nanoTime();
  dt_ms[t_d] = round(dt_ns[t_d]/1000);
  }
  else if(t_d == 0){
  time_t_0 = System.nanoTime();
  last_time = time_t_0;
  }
}



void concatenate_file(){
 int last_t_d = 0, t_d_overflow = 0; 
 
 StringList write_buffer;
 write_buffer = new StringList();
  
  if(!oworcat.button_activated){
  output = createWriter(type_file_name.variable_text);
  }
  else {
  if( null != loadStrings(type_file_name.variable_text) ){
    
  String[] file_buffer = loadStrings(type_file_name.variable_text);
  
  
  output = createWriter(type_file_name.variable_text);

  
    for(int j=0; j < file_buffer.length; j++){
     output.println(file_buffer[j]);
    }
  }
  else if( null == loadStrings(type_file_name.variable_text) ){
    //System.out.println("loadStrings returns null pointer");
    output = createWriter(type_file_name.variable_text);
  }
  
 }
 //add time stamp
 output.println(day() + "_" + month() + "_" + year() + ", " + hour() + ":" + minute() + ":" + second());
 output.println(freq_val + "\t" + (1.0/(freq_val*120))  + "\t" + k_val + "\t" + b_val + "\t" + eq_val);

 while(wfile.button_activated){
   
   if(t_d != last_t_d){
     if(last_t_d > t_d){t_d_overflow++;}
 write_buffer.append(( (t_d_overflow * ( width - 2*x_offset ) ) + t_d) + "\t" + t + "\t"
              + dt_ms[t_d] + "\t" + DAC1[t_d] + "\t" + A0[t_a] + "\t" 
              + amp + "\t" + phase_estimate + "\t"
              + centre + "\t" + round(peak_to_peak/2.0) + "\t"
              + A0_period_estimate); 
 /* 
 output.println( ( (t_d_overflow * ( width - 2*x_offset ) ) + t_d) + "\t" + t + "\t"
              + dt_ms[t_d] + "\t" + DAC1[t_d] + "\t" + A0[t_a] + "\t" 
              + amp + "\t" + phase_estimate + "\t"
              + centre + "\t" + round(peak_to_peak/2.0) + "\t"
              + A0_period_estimate);
              */
 last_t_d = t_d;
   } 
   
 }
 
 for(int i=0; i < write_buffer.size(); i++){
     output.println(write_buffer.get(i));
    }
 
 output.flush();  // Writes the remaining data to the file
 output.close();  // Finishes the file
 
}



void frequency_sweep(){
  
  boolean send_ok = false;
  float freq_start = 1.0, freq_end = 20.0, freq_increment = 1.0;
  int resp_count = 0, d_dev = 0, amp_dev = 0;
  int[] amp_compare = new int[12]; 
  freq_val = freq_start;
  
  val = int( ( (1.0/freq_val) * 42000000.0 ) / 120.0);
  // needs to be HALF of (1/freq) * (84 MHz/sample_num) for some reason  
  System.out.println( "val = " + val);
     //debug = saved;
     saved = "";
         
   val_in[0] = 0;   
   val_in[4] = (byte) ((val >> 24) & 0xFF);
   val_in[3] = (byte) ((val >> 16) & 0xFF);
   val_in[2] = (byte) ((val >> 8) & 0xFF);
   val_in[1] = (byte) (val & 0xFF);
   
   
   serialPort.write(val_in);/*
   System.out.println( " val_in[0] = " + val_in[0]
                     + " val_in[1] = " + val_in[1]
                     + " val_in[2] = " + val_in[2]
                     + " val_in[3] = " + val_in[3]
                     + " val_in[4] = " + val_in[4]);*/ //debug
   float t_unit = (1000.0 / (freq_val*120.0) );    
   time_axes("ms", t_unit);  
       
     
  while(fsweep.button_activated && freq_end >= freq_val && freq_start <= freq_val){
    
   if( 140 >= A0_period_estimate && 100 <= A0_period_estimate){
     amp_compare[resp_count] = amp;
        
     if( 1 < resp_count ){
       d_dev = 0;
       for( int i = 0; i < resp_count; i++){
         d_dev += abs(amp_compare[resp_count] - amp_compare[i]);
       }
      amp_dev = (d_dev/resp_count); 
     }
     if(20 <= amp_dev){resp_count ++;}
     else{resp_count = 0;}
   }
  else{resp_count = 0;} 
   
   if(12 <= resp_count){
     send_ok = true;
     resp_count = 0;   
   }
    
   if(send_ok){
     freq_val += freq_increment;
     val = int( ( (1.0/freq_val) * 42000000.0 ) / 120.0);
     // needs to be HALF of (1/freq) * (84 MHz/sample_num) for some reason  
     System.out.println( "val = " + val);
     //debug = saved;
     saved = "";
         
   val_in[0] = 0;   
   val_in[4] = (byte) ((val >> 24) & 0xFF);
   val_in[3] = (byte) ((val >> 16) & 0xFF);
   val_in[2] = (byte) ((val >> 8) & 0xFF);
   val_in[1] = (byte) (val & 0xFF);
   
   
   serialPort.write(val_in);/*
   System.out.println( " val_in[0] = " + val_in[0]
                     + " val_in[1] = " + val_in[1]
                     + " val_in[2] = " + val_in[2]
                     + " val_in[3] = " + val_in[3]
                     + " val_in[4] = " + val_in[4]);*/ //debug
   t_unit = (1000.0 / (freq_val*120.0) );    
   time_axes("ms", t_unit);
            
   send_ok = false;
 
   }
  }
}

void stress_of_time(){

  while(fsweep.button_activated){
    
  if(0 == t_d){  
   
    if(0 < stress_rate && 2048 > amp){
     val_in[0] = 4;
     serialPort.write(val_in[0]);
     type_set_strain.constant_text = (amp + 1) + " of 2048"; 
     System.out.println("val_in[0] = " + val_in[0]);
    }
    else if(0 > stress_rate && 0 < amp){
     val_in[0] = 2;
     serialPort.write(val_in[0]);
     type_set_strain.constant_text = (amp - 1) + " of 2048"; 
   }   
  }
 }
  
}

