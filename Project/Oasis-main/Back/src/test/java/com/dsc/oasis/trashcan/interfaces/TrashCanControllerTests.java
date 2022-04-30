package com.dsc.oasis.trashcan.interfaces;

import com.dsc.oasis.trashcan.application.TrashcanService;
import com.dsc.oasis.trashcan.domain.TrashCan;
import com.dsc.oasis.trashcan.dto.TrashCanResponse;
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
@WebMvcTest(TrashCanController.class)
public class TrashCanControllerTests {

    @Autowired
    private MockMvc mvc;

    @MockBean
    private TrashcanService trashcanService;

    @Test
    public void list() throws Exception {

        List<TrashCanResponse> trashCans=new ArrayList<>();

//        trashCans.add(TrashCan.builder().address("BUSAN").location("우리집 앞").trash_type("일반쓰레기").build());
//
//        given(trashcanService.getTrashCans()).willReturn(trashCans);
//
//        mvc.perform(get("/trashcans"))
//                .andExpect(status().isOk())
//                .andExpect(content().string(containsString("BUSAN")));
    }

    @Test
    public void detail() throws Exception {
        TrashCanResponse trashCanResponse=TrashCanResponse.builder().id(1L).build();

        given(trashcanService.getTrashCanById(1L)).willReturn(trashCanResponse);

        mvc.perform(get("/trashcans/1"))
                .andExpect(status().isOk())
                .andExpect(content().string(containsString("1")));
    }

}