package com.zerock.moijob;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
public class MoijobApplication {

    public static void main(String[] args) {
        SpringApplication.run(MoijobApplication.class, args);
    }

}
