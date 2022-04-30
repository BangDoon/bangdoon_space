package com.zerock.moijob.domain;

import com.zerock.moijob.dto.RecruitResponse;
import lombok.*;

import javax.persistence.*;
import java.util.List;

@Entity
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table(name= "tb1_recruit")
@ToString
public class Recruit {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String region;

    private String city;

    private String title;

    private String link;

    public RecruitResponse toResponse(){
        return RecruitResponse.builder().title(title).region(region).city(city).link(link).build();
    }

}
