In the end, our project was sidelined by some unknown issue that we were unable to resolve. We were unable to properly decode things that we sent over the radio, even though things worked fairly consistently in the ideal state - and you can see that our BER curves for these simulations look fairly sane.

Including are a few Ipython notebooks that show our tests. In Troubleshooting.ipynb, there are a bunch of plotted chirps that were sent and received via radio and/or SDR. They look fairly sane, and do not show that there were necessarily any obvious issues in transmitting and receiving, which did not help us narrow down our debugging process.

Using the same files we used for our project, we were able to decode the ISS packet from the lab, but when sending it, were not able to decode a packet that had been sent even though aprs.fi was able to decode it. We made the sampling rate of all these 48k, and that appears to be correct. 

