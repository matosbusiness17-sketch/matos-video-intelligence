Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("test_media/sample_speech.wav")

$script = @"
Welcome to the Matos Video Intelligence demonstration. 
In this episode, we are testing automated local AI video analysis.
Notice that the chart on the screen shows a 45 percent increase in video engagement.
The key to retaining short-form viewers is an irresistible hook in the first three seconds.
Our pipeline processes speech with Faster-Whisper and extracts keyframes with OpenCV.
Finally, we package everything cleanly for n8n workflow integration.
"@

$synth.Speak($script)
$synth.Dispose()
Write-Output "Speech generated successfully."
