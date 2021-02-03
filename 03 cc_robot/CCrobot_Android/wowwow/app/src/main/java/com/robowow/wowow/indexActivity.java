package com.robowow.wowow;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

import com.robowow.wowow.R;

public class indexActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_index);

        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                goMainActivity();
            }
        }, 1500);

    }
    public void goMainActivity(){
        Intent intent=new Intent (this, MainActivity.class);
        startActivity(intent);
        finish();
    }
}
