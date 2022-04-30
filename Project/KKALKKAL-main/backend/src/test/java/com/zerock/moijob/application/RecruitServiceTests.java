package com.zerock.moijob.application;

import com.zerock.moijob.domain.Recruit;
import com.zerock.moijob.domain.RecruitRepository;
import com.zerock.moijob.dto.RecruitResponse;
import org.junit.Before;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;
import static org.mockito.BDDMockito.given;

public class RecruitServiceTests {

    @Mock
    private RecruitRepository recruitRepository;

    private RecruitService recruitService;

    @Before
    public void setUp() {
        recruitService=new RecruitService(recruitRepository);
    }

    @Test
    public void getRecruits(){
        String region="Busan";
        String city="Busan";

        List<Recruit> mockRecruits=new ArrayList<>();

        mockRecruits.add(Recruit.builder().region(region).city(city).title("We want to recruit some people").link("http://www.recruit.com").build());

        given(recruitRepository.findAllByRegionContaining(region)).willReturn(mockRecruits);

        RecruitResponse recruitResponse=recruitService.getRecruits(region).get(0);

        assertThat(recruitResponse.getRegion(),is(region));
    }
}