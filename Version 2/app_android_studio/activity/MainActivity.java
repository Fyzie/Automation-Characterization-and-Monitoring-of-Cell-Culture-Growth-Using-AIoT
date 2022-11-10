package com.example.cellmonitor;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Button;
import android.widget.ImageView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    TextView temperature, humidity, light, white, door, confluency, countdown;
    String tempstatus, humistatus, lightstatus, whitestatus, doorstatus, confstatus, countstatus;
    Button yes, no;
    ImageView image;
    DatabaseReference dref;
    StorageReference sref;
    NotificationManagerCompat notificationManagerCompat;
    Notification notification;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O){
            NotificationChannel channel = new NotificationChannel("thecellmachine", "thecellmachine", NotificationManager.IMPORTANCE_DEFAULT);

            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }

        temperature = (TextView)  findViewById(R.id.tempdata);
        humidity = (TextView)  findViewById(R.id.humidata);
        light = (TextView)  findViewById(R.id.lightdata);
        white = (TextView)  findViewById(R.id.whitedata);
        door = (TextView)  findViewById(R.id.doordata);
        confluency = (TextView)  findViewById(R.id.confdata);
        countdown = (TextView)  findViewById(R.id.countdown);
        countdown.setText("Idle");
        yes = (Button) findViewById(R.id.button_yes);
        yes.setVisibility(View.INVISIBLE);
        no = (Button) findViewById(R.id.button_no);
        no.setVisibility(View.INVISIBLE);
        image = (ImageView) findViewById(R.id.cellimage);
        dref = FirebaseDatabase.getInstance().getReference();
        dref.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) { // when firebase data change
                tempstatus=dataSnapshot.child("Temperature").getValue().toString();
                temperature.setText(tempstatus);
                humistatus=dataSnapshot.child("Humidity").getValue().toString();
                humidity.setText(humistatus);
                lightstatus=dataSnapshot.child("Light").getValue().toString();
                light.setText(lightstatus);
                whitestatus=dataSnapshot.child("White").getValue().toString();
                white.setText(whitestatus);
                doorstatus=dataSnapshot.child("Door").getValue().toString();
                door.setText(doorstatus);
                sref = FirebaseStorage.getInstance().getReference("segmented_image.jpg");
                try {
                    File localfile = File.createTempFile("tempfile", ".jpg");
                    sref.getFile(localfile)
                            .addOnSuccessListener(new OnSuccessListener<FileDownloadTask.TaskSnapshot>() {
                                @Override
                                public void onSuccess(FileDownloadTask.TaskSnapshot taskSnapshot) {
                                    Bitmap bitmap = BitmapFactory.decodeFile(localfile.getAbsolutePath());
                                    image.setImageBitmap(bitmap);
                                }
                            }).addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {

                        }
                    });
                } catch (IOException e){
                    e.printStackTrace();
                }
                confstatus=dataSnapshot.child("Confluency").getValue().toString();
                confluency.setText(confstatus);
                countstatus=dataSnapshot.child("Countdown").getValue().toString();
                countdown.setText(countstatus);
                if (countstatus.equals("Idle")){
                    yes.setVisibility(View.INVISIBLE);
                    no.setVisibility(View.INVISIBLE);
                }
                else if (countstatus.equals("Transferring")){
                    yes.setVisibility(View.INVISIBLE);
                    no.setVisibility(View.INVISIBLE);
                }
                else if (countstatus.equals("Storing")){
                    yes.setVisibility(View.INVISIBLE);
                    no.setVisibility(View.INVISIBLE);
                }
                else if (countstatus.equals("Segmenting")){
                    yes.setVisibility(View.INVISIBLE);
                    no.setVisibility(View.INVISIBLE);
                }

                else{ // visible button when countdown started for user to choose 'YES'/'NO'
                    if (countstatus.equals("10")) {
                        // notify when new segmented cell received
                        Intent intentNotification = new Intent(MainActivity.this, MainActivity.class);
                        PendingIntent pendingIntent = PendingIntent.getActivity(MainActivity.this, 1, intentNotification, PendingIntent.FLAG_UPDATE_CURRENT);

                        NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this, "thecellmachine");
                        builder.setSmallIcon(R.drawable.ic_micro);
                        builder.setContentTitle("The Cell Machine");
                        builder.setContentText("Segmented confluency is available");
                        builder.setAutoCancel(true);
                        builder.setContentIntent(pendingIntent);

                        NotificationManagerCompat managerCompat = NotificationManagerCompat.from(MainActivity.this);
                        managerCompat.notify(1, builder.build());
                    }
                    yes.setVisibility(View.VISIBLE);
                    no.setVisibility(View.VISIBLE);
                }
            }
            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });
        yes.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                FirebaseDatabase database = FirebaseDatabase.getInstance();
                DatabaseReference myRef = database.getReference("Indicator");
                myRef.setValue(1); // force cell transfer
            }
        });
        no.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                FirebaseDatabase database = FirebaseDatabase.getInstance();
                DatabaseReference myRef = database.getReference("Indicator");
                myRef.setValue(3); // deny cell transfer
            }
        });

    }
}
