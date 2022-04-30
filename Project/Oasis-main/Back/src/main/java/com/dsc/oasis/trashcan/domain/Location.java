package com.dsc.oasis.trashcan.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

//단순 계산을 위한 Utility 클래스
@Getter
@Builder
@AllArgsConstructor
public class Location {
    private Double latitude;
    private Double longitude;
}
