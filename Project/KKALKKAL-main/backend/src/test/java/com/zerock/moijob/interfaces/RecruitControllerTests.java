package com.zerock.moijob.interfaces;

import com.zerock.moijob.application.RecruitService;
import com.zerock.moijob.dto.RecruitResponse;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.test.web.servlet.MockMvc;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;

@RunWith(SpringRunner.class)
@WebMvcTest(RecruitController.class)
public class RecruitControllerTests {

    @MockBean
    private RecruitService recruitService;

    @Autowired
    private MockMvc mvc;

    @Test
    public void list() throws Exception {
        List<RecruitResponse> mockRecruits=new ArrayList<>();
        String region="GyeongNam";
        String city="TongYoung";

        mockRecruits.add(RecruitResponse.builder().region(region).city(city).title("come on!").link("https://gyeongnam.com").build());

        given(recruitService.getRecruits(region)).willReturn(mockRecruits);

        mvc.perform(get("/recruits?region=GyeongNam"))
                .andExpect(status().isOk())
                .andExpect(content().string(containsString("\"region\":\"GyeongNam\"")));
    }

}