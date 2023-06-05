package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.myapplication.AudioTrack.ChirpEmitterBisccitAttempt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class RecognizeRoomExecutor  implements Runnable{
    RecognizeWindow recognizeWindow;

    String ip;

    private CyclicBarrier cyclicBarrier = new CyclicBarrier(1);

    public RecognizeRoomExecutor(RecognizeWindow recognizeWindow, String ip) {
        this.recognizeWindow = recognizeWindow;
        this.ip = ip;
    }

    @Override
    public void run() {
        AudioRecord audioRecord = createAudioRecord();
        int buffer_size = (int) (Globals.SAMPLE_RATE * Globals.RECORDING_INTERVAL * 7);
        short[] buffer = new short[buffer_size];

        List<short[]> listOfRecords = new ArrayList<>();

        audioRecord.startRecording();
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    cyclicBarrier.await();

                    audioRecord.read(buffer, 0, buffer_size);
                } catch (BrokenBarrierException e) {
                    throw new RuntimeException(e);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

            }
        }).start();

        ChirpEmitterBisccitAttempt.playSound(Globals.CHIRP_FREQUENCY, 7, cyclicBarrier);

        listOfRecords.add(Arrays.copyOf(buffer, buffer_size));
        audioRecord.stop();
        audioRecord.release();

        Room room = new Room(listOfRecords, "", "");
        String label = ServerCommunication.clasifyRoom(room, this.ip);

        this.recognizeWindow.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                recognizeWindow.get_label_text_view().setText(label);
            }
        });
    }

    private AudioRecord createAudioRecord() {

        if (ContextCompat.checkSelfPermission(recognizeWindow,
                Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {

            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(recognizeWindow,
                    Manifest.permission.RECORD_AUDIO)) {

                // Show an expanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.

            } else {

                // No explanation needed, we can request the permission.

                ActivityCompat.requestPermissions(recognizeWindow,
                        new String[]{Manifest.permission.RECORD_AUDIO},
                        1);

                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }

        }

        AudioRecord audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                44100,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                (int) (44100 * 0.1 * 2 * 100) // sampleRate*duration*2*repeats
        );
//        System.out.println(audioRecord.getBufferSizeInFrames());
        return audioRecord;
    }
}
