package com.dsc.oasis.trashcan.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.locationtech.jts.geom.Point;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TrashCanResponse {

    private Long id;

    private String address;

    private String location;

    private String trash_type;

    private Double latitude;

    private Double longitude;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Integer distance;

}
