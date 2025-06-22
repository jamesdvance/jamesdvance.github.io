## Observe, Orient, Decide, Act
Observe, Orent, DEcide, Act (OODA) is an acronym coined by an Airforce Colonel, John Boyd [1]. While Boyd used'OODA' to describe features of airplanes important to pilots' success, it's a useful way to describe video-based computer vision systems which understand a scenario and perform some action. Here is each step in detail: 

* Observe – Utilize all five senses, not just sight, to glean information about the situation.
The more information you have, the more accurate your perceptions will be.
* Orient - Understand the meaning of what you are observing. Analyze the information
available.
* Decide - Weigh options available and pick one. Changes to your decision can be
triggered by new information as you continue to learn more about the situation.
* Act - Carry out the decision. As you judge the effects of your action, you circle back to
“Observe” in a continuous loop.

In that framework, cameras allow us to observe, and computer vision allows us to orient. Eventually, 

## Modern Computer Vision and 'Physical AI'
Just as sight is crucial in how humans deal with situations in real life, video-based AI presents infinite possibilities for applications, provided computer vision systems can demonstrate sufficient situational understanding. Recently, VLMs (Visual Language Models) and MLLMs (Multi-Modal Large Language Models) have demonstrated much higher understanding than traditional computer vision tasks of detecting objects or classifying frames. Companies like SpotAI [2] have lept into action, using VLMs as video 'agents' to observe, understand and take some action based on this understanding. 

In the longer term, 'Physical AI' promises to let AI take action in the physical world. Again, vision is key, as Vision-Language-Action (VLA) Models like Google's Gemeni Robotics [3] tie together language input with physical tasks. 

For a personal project, there are hundreds of potential application of an 'OODA Box', or a small camera running a computer vision application that can interpret video in real time and take some action. These could be:

* Play a sound when squirrels enter the attic
* Alert when the parking space out front is empty
* Play a shushing sound when the baby starts moving in the crib

## A Simple Starter
A Raspberri Pi is small general-purpose computer small enough that it can be entitled to money. In this case, I use a Raspberri Pi 4 Model B, which has a hefty amount of RAM (8 GB) and many ports for actions. 


## ML Video Inference Options (Observe)

* RTP, WebRTC
* Edge inference vs remote

### Video Inference In Python


## Model Types (Orient)
CV Tasks and their general memory/compute requirements
* Object Detection, Classification, Motion Detection, Tracking
* Visual Question / Answering

## Control - Agents vs if/then (Decide)
* if/then logic
* classic control algorithms / SLAM
* agents

## Use Cases (Act)
* Speaker
* Record / store footage
* Deploy a drone with RCLink
* Move 
    * Simple direct control
    * 



### References
[1] [The Essense Of Winning And Losing](https://slightlyeastofnew.com/wp-content/uploads/2010/03/essence_of_winning_losing.pdf)
[2] [Spot AI](https://www.spot.ai/)
[3] [Gemini Visual Action Model](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)
[4] [Raspberri Pi 4 B Specs](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)

[4] [MotionEye](https://github.com/motioneye-project/motioneye)
[5] [RTP Streaming]()
[6] [WebRTC](https://webrtc.org/) 
[7] [RPi-Cam-Web-Interface](https://elinux.org/RPi-Cam-Web-Interface)
[] [VGGT](https://github.com/facebookresearch/vggt)
[VLM Models For Jetson](https://www.jetson-ai-lab.com/models.html)