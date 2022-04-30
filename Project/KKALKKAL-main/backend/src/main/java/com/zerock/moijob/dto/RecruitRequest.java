package com.zerock.moijob.dto;

import com.zerock.moijob.domain.Recruit;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecruitRequest {


    private String region;

    private String city;

    private String title;

    private String link;

    public Recruit toEntity(){
       return Recruit.builder().title(title).region(region).city(city).link(link).build();
    }
}
