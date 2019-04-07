package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.ByteArrayOutputStream;
import java.io.FileDescriptor;
import java.io.IOException;

public class Options extends Activity {




    public  void openRealtime(View view){
        Intent i=new Intent(getApplicationContext(),StylizeActivity.class);
        startActivity(i);


    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.oplayout);

     Button button=(Button)findViewById(R.id.button2);
     button.setOnClickListener(new View.OnClickListener() {
         @Override
         public void onClick(View view) {
             Intent i=getPackageManager().getLaunchIntentForPackage("com.android.example.renderscript_neuralnet");
             startActivity(i);
         }
     });
    }



    }






