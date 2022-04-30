package com.zerock.moijob.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecruitResponse {

    private Long id;

    private String region;

    private String city;

    private String title;

    private String link;
}
