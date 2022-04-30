package com.dsc.oasis.trashcan.dto;

import com.dsc.oasis.trashcan.domain.Location;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TrashCanRequest {

    private Double latitude;

    private Double longitude;

    public Location toEntity(){
        return Location.builder()
                .latitude(latitude)
                .longitude(longitude)
                .build();
    }
}
