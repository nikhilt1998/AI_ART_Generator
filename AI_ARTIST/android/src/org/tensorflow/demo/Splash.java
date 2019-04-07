package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Window;
import android.view.WindowManager;

public class Splash extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.splashx);


        LogoLauncher logoLauncher=new LogoLauncher();
        logoLauncher.start();


    }



    class  LogoLauncher extends Thread{

        public void run(){

            try {
                sleep(4000);

            }catch(InterruptedException e){
                e.printStackTrace();
            }

            Intent intent;
            intent = new Intent(Splash.this,Options.class);
            startActivity(intent);
            Splash.this.finish();
        }


    }


}
