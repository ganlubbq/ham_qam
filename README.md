Demo.ipynb contains a demonstration and detailed explanation of the system.

In the end, we were unable to demonstrate a working communication system using the handheld radios due to an unknown issue receiving data.
We were unable to properly decode things that we sent over the radio, even though things worked fairly consistently in the ideal state - and you can see that our BER curves for these simulations look fairly sane.

Included are a few Ipython notebooks that show our tests. In Troubleshooting.ipynb, there are plots of chirps that were sent and received via radio and/or SDR. 
They look fairly sane, and do not show that there were necessarily any obvious issues in transmitting and receiving, which did not help us narrow down our debugging process.

Detailed troubleshooting steps:
- We were able to successfully send APRS position updates that were logged on
  aprs.fi using the same system we used to transmit our QAM modulated data.

- Decoding the sound file sent to the radio for transmission resulted in 0
  symbol errors. This combined with our successful ability to send APRS packets
  led us to believe that the problem was in our receiver system.

- We attempted to send our QAM signal at varying symbol rates (even as slow as 1
  symbol per second) so we could visually identify symbol transitions. When
  demodulating this output we were able to observe the correct number of symbol 
  transisions, as these amplitude of the signal was significantly larger than
  the silence transmitted before it; however demodulated waveforms themselves
  had no visually or mathematically significant correlation with the expected
  signal.

- We observed the same result when attempting to record APRS packets which were
  successfully transmitted, decoded, and logged on aprs.fi but were unable to
  detect any packets.

- In order to test the APRS decoding we referred to the previous lab document
  and were able to successfully demodulate and decode an APRS packet from the
  ISS.
  
- We attempted to receive using both the UV-5R and the SDR via GQRX's audio recording
  functionality. In all cases we were able to hear both the transmitted and
  received audio. Receiving FM radio such as from 94.1 KPFA sounded correct.

- In order to confirm that the transmission was not saturating the recording
  devices by transmitting in the same room we attempted transmitting and
  receiving at different volume and gain settings at varying distances.

- All processing was performed at 48 kHz consistently throughout the system in
  all configurations.

- We noticed a number of peaks in the spectrum of the signal and tested a number
  of different FIR filters to isolate these frequencies. We were unable to
  identify the prefix or any symbols.

- We attempted to shape the pulses with Hann windowed sincs, however this also
  didn't produce any decodable output, or even output that was visually similar
  to the transmitted data.
