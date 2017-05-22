ESCAPE
======

Echo SCraper and ClAssifier of PErsons: A novel tool to facilitate using voice-controlled devices for research 


## Usage


Before this script can be used you have to copy and paste a cURL from your account. Instructions given for Chrome.

1. Go to [http://alexa.amazon.co.uk/](http://alexa.amazon.co.uk/) and sign in to your account
2. Navigate to your History: Menu > Settings > General > History (you can access the cookie from other locations but it's helpful to know where this data is for sanity checking downloads)
3. Open Developer Tools: View > Developer > Developer Tools
4. Open the Network tab in the Developer Tools panel
![Alt](https://image.ibb.co/e1TL0a/Network_Tab.png "Network Tab")
5. Click on any interaction from your history, you should see some items appear in the Network tab
6. Look for the item in the Network tab that starts with `activity-dialog-items` (it should be top)
7. Right click on this item > Copy >  Copy as cURL
8. Now paste this into a blank plain text file and save it 

## ESCAPE

The easiest way to use ESCAPE is to use the ESCAPE.py script. To do this, simply run the script from the same location as the cURL file, or us the -c flag to point to the file. The script will then download all the data, and proceed with KL classification and then HMM modelling of the audio files that have been manually tagged. The script will generate a PCA plot of the first three principal components of the manually tagged data.