/* This file is part of VoltDB.
 * Copyright (C) 2008-2018 VoltDB Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.voltcore.agreement;

public enum ArbitrationStrategy {
    NO_QUARTER {
        @Override
        public <R, P> R accept(Visitor<R, P> vtor, P param) {
            return vtor.visitNoQuarter(param);
        }
    },
    MATCHING_CARDINALITY {
        @Override
        public <R, P> R accept(Visitor<R, P> vtor, P param) {
            return vtor.visitMatchingCardinality(param);
        }
    };

    public interface Visitor<R,P> {
        R visitNoQuarter(P param);
        R visitMatchingCardinality(P param);
    }

    public abstract <R,P> R accept(Visitor<R,P> vtor, P param);
}